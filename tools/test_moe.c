/**
 * test_moe.c — Test MoE forward for a single layer
 * Loads one layer's MoE weights, runs 4 tokens through router + experts
 */
#include "wubu_moe.h"
#include "gguf_reader.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Load one layer's MoE weights from GGUF
static int load_moe_layer(const char *path, int layer_idx, moe_weights_t *moe) {
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 0;

    memset(moe, 0, sizeof(*moe));
    char name[256];

    // Router
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", layer_idx);
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (t) {
        moe->ffn_gate_inp = malloc(D_MODEL * N_EXPERTS * sizeof(float));
        if (gguf_read_tensor_f32(ctx, t, moe->ffn_gate_inp, D_MODEL * N_EXPERTS) <= 0)
            { fprintf(stderr, "Failed: %s\n", name); return 0; }
    } else {
        fprintf(stderr, "Missing: %s\n", name); return 0;
    }

    // Expert gate
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (t) {
        int64_t n = (int64_t)D_MODEL * D_FF * N_EXPERTS;
        moe->ffn_gate_exps = malloc(n * sizeof(float));
        if (gguf_read_tensor_f32(ctx, t, moe->ffn_gate_exps, n) <= 0)
            { fprintf(stderr, "Failed: %s\n", name); return 0; }
    } else { fprintf(stderr, "Missing: %s\n", name); return 0; }

    // Expert up
    snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (t) {
        int64_t n = (int64_t)D_MODEL * D_FF * N_EXPERTS;
        moe->ffn_up_exps = malloc(n * sizeof(float));
        if (gguf_read_tensor_f32(ctx, t, moe->ffn_up_exps, n) <= 0)
            { fprintf(stderr, "Failed: %s\n", name); return 0; }
    } else { fprintf(stderr, "Missing: %s\n", name); return 0; }

    // Expert down
    snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (t) {
        int64_t n = (int64_t)D_FF * D_MODEL * N_EXPERTS;
        moe->ffn_down_exps = malloc(n * sizeof(float));
        if (gguf_read_tensor_f32(ctx, t, moe->ffn_down_exps, n) <= 0)
            { fprintf(stderr, "Failed: %s\n", name); return 0; }
    } else { fprintf(stderr, "Missing: %s\n", name); return 0; }

    // Shared expert gate
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (t) {
        moe->ffn_gate_shexp = malloc(D_MODEL * SHARED_D_FF * sizeof(float));
        if (gguf_read_tensor_f32(ctx, t, moe->ffn_gate_shexp, D_MODEL * SHARED_D_FF) <= 0)
            { fprintf(stderr, "Failed: %s\n", name); return 0; }
    } else { fprintf(stderr, "Missing: %s\n", name); return 0; }

    // Shared expert up
    snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (t) {
        moe->ffn_up_shexp = malloc(D_MODEL * SHARED_D_FF * sizeof(float));
        if (gguf_read_tensor_f32(ctx, t, moe->ffn_up_shexp, D_MODEL * SHARED_D_FF) <= 0)
            { fprintf(stderr, "Failed: %s\n", name); return 0; }
    } else { fprintf(stderr, "Missing: %s\n", name); return 0; }

    // Shared expert down
    snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", layer_idx);
    t = gguf_find_tensor(ctx, name);
    if (t) {
        moe->ffn_down_shexp = malloc(SHARED_D_FF * D_MODEL * sizeof(float));
        if (gguf_read_tensor_f32(ctx, t, moe->ffn_down_shexp, SHARED_D_FF * D_MODEL) <= 0)
            { fprintf(stderr, "Failed: %s\n", name); return 0; }
    } else { fprintf(stderr, "Missing: %s\n", name); return 0; }

    gguf_close(ctx);
    moe->loaded = true;
    return 1;
}

static void free_moe(moe_weights_t *moe) {
    free(moe->ffn_gate_inp);
    free(moe->ffn_gate_exps);
    free(moe->ffn_up_exps);
    free(moe->ffn_down_exps);
    free(moe->ffn_gate_shexp);
    free(moe->ffn_up_shexp);
    free(moe->ffn_down_shexp);
    free(moe->ffn_gate_inp_shexp);
    memset(moe, 0, sizeof(*moe));
}

int main(void) {
    const char *path = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    int layer = 0;  // Test layer 0
    
    printf("=== MoE Test: Layer %d ===\n", layer);
    
    moe_weights_t moe;
    double t0 = now_sec();
    if (!load_moe_layer(path, layer, &moe)) {
        fprintf(stderr, "Failed to load MoE weights\n");
        return 1;
    }
    printf("  Loaded MoE weights in %.2fs\n", now_sec() - t0);
    
    // Test input: 4 tokens, random embeddings
    int B = 1, T = 4, N = B * T;
    float x[4 * 2048];
    for (int i = 0; i < N * D_MODEL; i++)
        x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    
    // Router test
    float *scores = (float *)malloc(N * N_EXPERTS * sizeof(float));
    t0 = now_sec();
    wubu_moe_router(x, B, T, moe.ffn_gate_inp, scores);
    printf("  Router: %.3fms\n", (now_sec() - t0) * 1000);
    
    // Print top-5 expert scores for first token
    int top5[5];
    float top5v[5];
    for (int k = 0; k < 5; k++) {
        float best = -1e30f;
        int best_i = -1;
        for (int e = 0; e < N_EXPERTS; e++) {
            int used = 0;
            for (int pk = 0; pk < k; pk++) if (top5[pk] == e) { used = 1; break; }
            if (!used && scores[e] > best) { best = scores[e]; best_i = e; }
        }
        top5[k] = best_i;
        top5v[k] = best;
    }
    printf("  Token 0 top-5 experts:");
    for (int k = 0; k < 5; k++) printf(" [%d]=%.4f", top5[k], top5v[k]);
    printf("\n");
    
    // Full MoE forward
    float *output = (float *)malloc(N * D_MODEL * sizeof(float));
    t0 = now_sec();
    wubu_moe_forward(x, B, T, &moe, output, NULL);
    double t_moe = now_sec() - t0;
    printf("  MoE forward: %.3fms (%.1f tok/s)\n", t_moe * 1000, N / t_moe);
    
    // Check output stats
    float min_v = 1e30, max_v = -1e30;
    for (int i = 0; i < N * D_MODEL; i++) {
        if (output[i] < min_v) min_v = output[i];
        if (output[i] > max_v) max_v = output[i];
    }
    printf("  Output range: [%.4f, %.4f]\n", min_v, max_v);
    
    // Check for NaN
    int nan_count = 0;
    for (int i = 0; i < N * D_MODEL; i++)
        if (isnan(output[i])) nan_count++;
    printf("  NaN count: %d\n", nan_count);
    
    free(scores);
    free(output);
    free_moe(&moe);
    
    printf("\n=== MoE Test PASS ===\n");
    return 0;
}
