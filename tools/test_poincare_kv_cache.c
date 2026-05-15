/**
 * test_poincare_kv_cache.c — Unit test for Poincaré GQA hyperbolic KV cache.
 */
#include "wubu_ssm.h"
#include "wubu_poincare_gqa.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float frand(void) {
    static unsigned int seed = 12345;
    seed = seed * 1103515245 + 12345;
    return (float)(seed & 0x7fffffff) / (float)0x7fffffff;
}

static void init_weights_2d(float *w, int rows, int cols, float scale) {
    for (int i = 0; i < rows * cols; i++)
        w[i] = (frand() - 0.5f) * 2.0f * scale;
}

int main(void) {
    const int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;
    const int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;
    const float R = 10.0f;
    int pass = 1;

    printf("=== Poincaré GQA Hyperbolic KV Cache Test ===\n\n");

    // Setup weights
    gqa_layer_weights w;
    memset(&w, 0, sizeof(w));
    w.attn_q_weight = (float *)malloc(D_MODEL * q_dim * 2 * sizeof(float));
    w.attn_k_weight = (float *)malloc(D_MODEL * kv_dim * sizeof(float));
    w.attn_v_weight = (float *)malloc(D_MODEL * kv_dim * sizeof(float));
    w.attn_output_weight = (float *)malloc(q_dim * D_MODEL * sizeof(float));
    w.attn_q_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
    w.attn_k_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));

    init_weights_2d(w.attn_q_weight, D_MODEL, q_dim * 2, 0.01f);
    init_weights_2d(w.attn_k_weight, D_MODEL, kv_dim, 0.01f);
    init_weights_2d(w.attn_v_weight, D_MODEL, kv_dim, 0.01f);
    init_weights_2d(w.attn_output_weight, q_dim, D_MODEL, 0.01f);
    for (int i = 0; i < GQA_HEAD_DIM; i++) {
        w.attn_q_norm_weight[i] = 1.0f;
        w.attn_k_norm_weight[i] = 1.0f;
    }

    // ===== Test 1: Cache lifecycle =====
    printf("Test 1: Cache init/resize/free...\n");
    {
        poincare_kv_cache_t cache;
        poincare_kv_cache_init(&cache, 4);
        if (!cache.K_ball_cached || !cache.V_ball_cached || cache.max_T != 4) {
            printf("  FAIL: init\n"); pass = 0; goto done;
        }
        poincare_kv_cache_resize(&cache, 32);
        if (cache.max_T != 32) { printf("  FAIL: resize\n"); pass = 0; goto done; }
        poincare_kv_cache_free(&cache);
        printf("  PASS\n");
    }

    // ===== Test 2: Prefill 2 then cache-forward 1 =====
    printf("Test 2: Prefill+Cache step...\n");
    {
        int B = 1, T_pre = 2, T_total = 3;
        int N_pre = B * T_pre, N_total = B * T_total;

        float *x = (float *)malloc(N_total * D_MODEL * sizeof(float));
        for (int i = 0; i < N_total * D_MODEL; i++)
            x[i] = (frand() - 0.5f) * 0.1f;

        // Reference: all 3 at once
        float *out_ref = (float *)calloc(N_total * D_MODEL, sizeof(float));
        poincare_gqa_fwd_save_t save_ref;
        memset(&save_ref, 0, sizeof(save_ref));
        save_ref.cache = NULL;
        wubu_poincare_gqa_forward_save(x, B, T_total, &w, R, out_ref, &save_ref);

        // Prefill 2 tokens, save their K/V_ball
        float *out_pre = (float *)calloc(N_pre * D_MODEL, sizeof(float));
        poincare_gqa_fwd_save_t save_pre;
        memset(&save_pre, 0, sizeof(save_pre));
        save_pre.cache = NULL;
        save_pre.Q_ball = (float *)malloc(N_pre * q_dim * sizeof(float));
        save_pre.K_ball = (float *)malloc(N_pre * kv_dim * sizeof(float));
        save_pre.V_ball = (float *)malloc(N_pre * kv_dim * sizeof(float));
        save_pre.Q_norm = NULL; save_pre.Q_raw = NULL;
        save_pre.K_norm = NULL; save_pre.K_raw = NULL;
        save_pre.V = NULL; save_pre.gate = NULL;
        save_pre.gate_sig = NULL; save_pre.attn_out_pre_gate = NULL;
        wubu_poincare_gqa_forward_save(x, B, T_pre, &w, R, out_pre, &save_pre);

        // Verify prefill matches reference
        float diff_pre = 0;
        for (int i = 0; i < N_pre * D_MODEL; i++) {
            float d = fabsf(out_pre[i] - out_ref[i]);
            if (d > diff_pre) diff_pre = d;
        }
        printf("  Prefill vs ref: %.2e\n", diff_pre);
        if (diff_pre > 1e-6f) { printf("  FAIL\n"); pass = 0; goto done; }

        // Setup cache with prefill results
        poincare_kv_cache_t cache;
        poincare_kv_cache_init(&cache, 4);
        memcpy(cache.K_ball_cached, save_pre.K_ball, (int64_t)N_pre * kv_dim * sizeof(float));
        memcpy(cache.V_ball_cached, save_pre.V_ball, (int64_t)N_pre * kv_dim * sizeof(float));
        cache.current_T = T_pre;

        // Cache-forward the 3rd token
        float *x_new = x + (int64_t)N_pre * D_MODEL;
        float *out_new = (float *)calloc(D_MODEL, sizeof(float));
        poincare_gqa_fwd_save_t save_new;
        memset(&save_new, 0, sizeof(save_new));
        save_new.cache = &cache;
        save_new.Q_ball = (float *)malloc(B * 1 * q_dim * sizeof(float));
        save_new.K_ball = (float *)malloc(B * 1 * kv_dim * sizeof(float));
        save_new.V_ball = (float *)malloc(B * 1 * kv_dim * sizeof(float));
        save_new.Q_norm = NULL; save_new.Q_raw = NULL;
        save_new.K_norm = NULL; save_new.K_raw = NULL;
        save_new.V = NULL; save_new.gate = NULL;
        save_new.gate_sig = NULL; save_new.attn_out_pre_gate = NULL;
        wubu_poincare_gqa_forward_save(x_new, B, 1, &w, R, out_new, &save_new);

        // Compare 3rd token
        float diff_t3 = 0;
        for (int i = 0; i < D_MODEL; i++) {
            float d = fabsf(out_new[i] - out_ref[(T_pre * D_MODEL) + i]);
            if (d > diff_t3) diff_t3 = d;
        }
        printf("  Token 3 (cached) vs ref: %.2e\n", diff_t3);
        printf("  Cache: T=%d/%d\n", cache.current_T, cache.max_T);
        if (diff_t3 > 1e-4f || cache.current_T != T_total) {
            printf("  FAIL\n"); pass = 0;
        }
        if (diff_t3 <= 1e-4f && cache.current_T == T_total) printf("  PASS\n");

        free(x); free(out_ref); free(out_pre); free(out_new);
        free(save_pre.Q_ball); free(save_pre.K_ball); free(save_pre.V_ball);
        free(save_new.Q_ball); free(save_new.K_ball); free(save_new.V_ball);
        poincare_kv_cache_free(&cache);
    }

    // ===== Test 3: Incremental T=1 with auto-grow =====
    printf("Test 3: Incremental T=1 * 4 steps auto-grow...\n");
    {
        int B = 1, steps = 4;
        poincare_kv_cache_t cache;
        poincare_kv_cache_init(&cache, 1);

        float *x = (float *)malloc(steps * D_MODEL * sizeof(float));
        for (int i = 0; i < steps * D_MODEL; i++)
            x[i] = (frand() - 0.5f) * 0.1f;

        // Reference
        float *out_ref = (float *)calloc(steps * D_MODEL, sizeof(float));
        poincare_gqa_fwd_save_t save_ref;
        memset(&save_ref, 0, sizeof(save_ref));
        save_ref.cache = NULL;
        wubu_poincare_gqa_forward_save(x, B, steps, &w, R, out_ref, &save_ref);

        // Incremental
        float *out_inc = (float *)calloc(steps * D_MODEL, sizeof(float));
        for (int s = 0; s < steps; s++) {
            float *xs = x + (int64_t)s * D_MODEL;
            float *os = out_inc + (int64_t)s * D_MODEL;

            poincare_gqa_fwd_save_t save_s;
            memset(&save_s, 0, sizeof(save_s));
            save_s.cache = &cache;
            save_s.Q_ball = (float *)malloc(B * 1 * q_dim * sizeof(float));
            save_s.K_ball = (float *)malloc(B * 1 * kv_dim * sizeof(float));
            save_s.V_ball = (float *)malloc(B * 1 * kv_dim * sizeof(float));
            save_s.Q_norm = NULL; save_s.Q_raw = NULL;
            save_s.K_norm = NULL; save_s.K_raw = NULL;
            save_s.V = NULL; save_s.gate = NULL;
            save_s.gate_sig = NULL; save_s.attn_out_pre_gate = NULL;

            wubu_poincare_gqa_forward_save(xs, B, 1, &w, R, os, &save_s);

            free(save_s.Q_ball);
            free(save_s.K_ball);
            free(save_s.V_ball);
        }

        float max_diff = 0;
        for (int i = 0; i < steps * D_MODEL; i++) {
            float d = fabsf(out_inc[i] - out_ref[i]);
            if (d > max_diff) max_diff = d;
        }
        printf("  Max diff: %.2e\n", max_diff);
        printf("  Cache: T=%d/%d\n", cache.current_T, cache.max_T);
        if (max_diff > 1e-4f || cache.current_T != steps) {
            printf("  FAIL\n"); pass = 0;
        }
        if (max_diff <= 1e-4f && cache.current_T == steps) printf("  PASS\n");

        free(x); free(out_ref); free(out_inc);
        poincare_kv_cache_free(&cache);
    }

done:
    printf("\n=== %s ===\n", pass ? "ALL TESTS PASSED" : "FAILED");
    free(w.attn_q_weight); free(w.attn_k_weight);
    free(w.attn_v_weight); free(w.attn_output_weight);
    free(w.attn_q_norm_weight); free(w.attn_k_norm_weight);
    return pass ? 0 : 1;
}
