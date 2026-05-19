#include "wubu_ssm.h"
#include "wubu_mobius.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================
// Test: SSM forward pass against Python reference
// ============================================================

static int test_ssm_forward() {
    printf("=== SSM Forward Pass Test ===\n");

    FILE *f = fopen("data/ssm_test_vectors.bin", "rb");
    if (!f) {
        printf("FAIL: Cannot open data/ssm_test_vectors.bin\n");
        printf("  Run: python3 tools/gen_test_vectors.py\n");
        return 1;
    }

    int B, T, D, V;
    fread(&B, sizeof(int), 1, f);
    fread(&T, sizeof(int), 1, f);
    fread(&D, sizeof(int), 1, f);
    fread(&V, sizeof(int), 1, f);

    printf("  B=%d, T=%d, D=%d, V=%d\n", B, T, D, V);

    int N = B * T;
    float *x = (float *)malloc(N * D * sizeof(float));
    fread(x, sizeof(float), N * D, f);

    ssm_layer_weights w;
    int qkv_dim = KEY_DIM * 2 + VALUE_DIM;
    w.attn_qkv_weight = (float *)malloc(D * qkv_dim * sizeof(float));
    fread(w.attn_qkv_weight, sizeof(float), D * qkv_dim, f);

    w.attn_gate_weight = (float *)malloc(D * VALUE_DIM * sizeof(float));
    fread(w.attn_gate_weight, sizeof(float), D * VALUE_DIM, f);

    w.ssm_beta_weight = (float *)malloc(D * DT_RANK * sizeof(float));
    fread(w.ssm_beta_weight, sizeof(float), D * DT_RANK, f);

    w.ssm_alpha_weight = (float *)malloc(D * DT_RANK * sizeof(float));
    fread(w.ssm_alpha_weight, sizeof(float), D * DT_RANK, f);

    w.ssm_dt_bias = (float *)malloc(DT_RANK * sizeof(float));
    fread(w.ssm_dt_bias, sizeof(float), DT_RANK, f);

    w.ssm_a = (float *)malloc(DT_RANK * sizeof(float));
    fread(w.ssm_a, sizeof(float), DT_RANK, f);

    w.ssm_conv1d_weight = (float *)malloc(CONV_KERNEL * CONV_DIM * sizeof(float));
    fread(w.ssm_conv1d_weight, sizeof(float), CONV_KERNEL * CONV_DIM, f);

    w.ssm_norm_weight = (float *)malloc(SSM_D_STATE * sizeof(float));
    fread(w.ssm_norm_weight, sizeof(float), SSM_D_STATE, f);

    w.ssm_out_weight = (float *)malloc(VALUE_DIM * D * sizeof(float));
    fread(w.ssm_out_weight, sizeof(float), VALUE_DIM * D, f);

    float *expected = (float *)malloc(N * D * sizeof(float));
    fread(expected, sizeof(float), N * D, f);
    fclose(f);

    float *ssm_state = (float *)calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));
    float *conv_state = (float *)calloc(B * (CONV_KERNEL - 1) * CONV_DIM, sizeof(float));

    float *output = (float *)malloc(N * D * sizeof(float));
    wubu_ssm_forward(x, B, T, &w, ssm_state, conv_state, output);

    float max_diff = 0.0f, sum_diff = 0.0f;
    int max_diff_idx = 0;
    for (int i = 0; i < N * D; i++) {
        float diff = fabsf(output[i] - expected[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        sum_diff += diff;
    }
    float avg_diff = sum_diff / (N * D);

    printf("  Max diff: %e (at idx %d)\n", max_diff, max_diff_idx);
    printf("  Avg diff: %e\n", avg_diff);
    printf("  Output[0]: C=%f Py=%f\n", output[0], expected[0]);
    printf("  Output[100]: C=%f Py=%f\n", output[100], expected[100]);

    free(x);
    free(w.attn_qkv_weight);
    free(w.attn_gate_weight);
    free(w.ssm_beta_weight);
    free(w.ssm_alpha_weight);
    free(w.ssm_dt_bias);
    free(w.ssm_a);
    free(w.ssm_conv1d_weight);
    free(w.ssm_norm_weight);
    free(w.ssm_out_weight);
    free(expected);
    free(ssm_state);
    free(conv_state);
    free(output);

    float tolerance = 3e-2f;  // 3% — numerical diff from float32 C vs Python float64 ref
    if (max_diff <= tolerance) {
        printf("  PASS (max_diff=%.2e <= %.0e)\n", max_diff, tolerance);
        return 0;
    } else {
        printf("  FAIL (max_diff=%.2e > %.0e)\n", max_diff, tolerance);
        return 1;
    }
}

// ============================================================
// Test: GQA forward pass
// ============================================================

static int test_gqa_forward() {
    printf("\n=== GQA Forward Pass Test ===\n");

    FILE *f = fopen("data/gqa_test_vectors.bin", "rb");
    if (!f) {
        printf("FAIL: Cannot open data/gqa_test_vectors.bin\n");
        printf("  Run: python3 tools/gen_test_vectors.py\n");
        return 1;
    }

    int B, T, D, Q;
    fread(&B, sizeof(int), 1, f);
    fread(&T, sizeof(int), 1, f);
    fread(&D, sizeof(int), 1, f);
    fread(&Q, sizeof(int), 1, f);

    printf("  B=%d, T=%d, D=%d, Q=%d\n", B, T, D, Q);

    int N = B * T;
    float *x = (float *)malloc(N * D * sizeof(float));
    fread(x, sizeof(float), N * D, f);

    gqa_layer_weights w;
    int q_dim = 16 * 256;
    int kv_dim = 2 * 256;

    w.attn_q_weight = (float *)malloc(D * q_dim * 2 * sizeof(float));
    fread(w.attn_q_weight, sizeof(float), D * q_dim * 2, f);

    w.attn_k_weight = (float *)malloc(D * kv_dim * sizeof(float));
    fread(w.attn_k_weight, sizeof(float), D * kv_dim, f);

    w.attn_v_weight = (float *)malloc(D * kv_dim * sizeof(float));
    fread(w.attn_v_weight, sizeof(float), D * kv_dim, f);

    w.attn_output_weight = (float *)malloc(q_dim * D * sizeof(float));
    fread(w.attn_output_weight, sizeof(float), q_dim * D, f);

    w.attn_q_norm_weight = (float *)malloc(256 * sizeof(float));
    fread(w.attn_q_norm_weight, sizeof(float), 256, f);

    w.attn_k_norm_weight = (float *)malloc(256 * sizeof(float));
    fread(w.attn_k_norm_weight, sizeof(float), 256, f);

    float *expected = (float *)malloc(N * D * sizeof(float));
    fread(expected, sizeof(float), N * D, f);
    fclose(f);

    float *output = (float *)malloc(N * D * sizeof(float));
    wubu_gqa_forward(x, B, T, &w, output, NULL, NULL, 0, NULL, NULL);

    float max_diff = 0.0f, sum_diff = 0.0f;
    for (int i = 0; i < N * D; i++) {
        float diff = fabsf(output[i] - expected[i]);
        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;
    }

    printf("  Max diff: %e\n", max_diff);
    printf("  Avg diff: %e\n", sum_diff / (N * D));

    free(x);
    free(w.attn_q_weight);
    free(w.attn_k_weight);
    free(w.attn_v_weight);
    free(w.attn_output_weight);
    free(w.attn_q_norm_weight);
    free(w.attn_k_norm_weight);
    free(expected);
    free(output);

    float tolerance = 3e-2f;  // 3% — numerical diff from float32 C vs Python float64 ref
    if (max_diff <= tolerance) {
        printf("  PASS (max_diff=%.2e <= %.0e)\n", max_diff, tolerance);
        return 0;
    } else {
        printf("  FAIL (max_diff=%.2e > %.0e)\n", max_diff, tolerance);
        return 1;
    }
}

// ============================================================
// Test: Layer type classification
// ============================================================

static int test_layer_types() {
    printf("\n=== Layer Type Classification Test ===\n");
    
    int ssm_count = 0, gqa_count = 0;
    for (int i = 0; i < 40; i++) {
        if (wubu_is_ssm_layer(i)) {
            ssm_count++;
        } else {
            gqa_count++;
        }
    }
    printf("  SSM layers: %d (expected 30)\n", ssm_count);
    printf("  GQA layers: %d (expected 10)\n", gqa_count);
    
    if (ssm_count == 30 && gqa_count == 10) {
        printf("  PASS\n");
        return 0;
    } else {
        printf("  FAIL\n");
        return 1;
    }
}

int main() {
    int failures = 0;
    failures += test_ssm_forward();
    failures += test_gqa_forward();
    failures += test_layer_types();
    
    printf("\n=== %s ===\n", failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED");
    return failures;
}
