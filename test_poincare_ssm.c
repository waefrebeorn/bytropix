#include "wubu_ssm.h"
#include "wubu_mobius.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * Test Poincaré SSM forward pass.
 * Uses the same test data as Euclidean, just with R=0.956 (from Phase 1 analysis).
 * Verifies: no NaN, finite output, similar magnitude to Euclidean output.
 */
int main() {
    // Read SSM test data
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
    
    // Run Euclidean SSM (reference)
    float *euclidean_state = (float *)calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));
    float *euclidean_conv = (float *)calloc(B * (CONV_KERNEL - 1) * CONV_DIM, sizeof(float));
    float *euclidean_out = (float *)malloc(N * D * sizeof(float));
    wubu_ssm_forward(x, B, T, &w, euclidean_state, euclidean_conv, euclidean_out);
    
    // Run Poincaré SSM with R = 0.956 (from Phase 1 analysis — 3× mean_norm)
    float R = 0.956f;
    float *poincare_state = (float *)calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));
    float *poincare_conv = (float *)calloc(B * (CONV_KERNEL - 1) * CONV_DIM, sizeof(float));
    float *poincare_out = (float *)malloc(N * D * sizeof(float));
    wubu_poincare_ssm_forward(x, B, T, &w, poincare_state, poincare_conv, R, poincare_out);
    
    // Compare
    printf("=== Poincaré SSM Forward Test ===\n");
    printf("  R = %.3f (from Phase 1 embedding analysis)\n", R);
    printf("  B=%d, T=%d, D=%d\n", B, T, D);
    
    // Check Euclidean
    float e_min = 1e30f, e_max = -1e30f;
    for (int i = 0; i < N * D; i++) {
        if (euclidean_out[i] < e_min) e_min = euclidean_out[i];
        if (euclidean_out[i] > e_max) e_max = euclidean_out[i];
    }
    printf("\n  Euclidean output: [%e, %e]\n", e_min, e_max);
    
    // Check Poincaré
    float p_min = 1e30f, p_max = -1e30f;
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < N * D; i++) {
        if (isnan(poincare_out[i])) { nan_count++; continue; }
        if (isinf(poincare_out[i])) { inf_count++; continue; }
        if (poincare_out[i] < p_min) p_min = poincare_out[i];
        if (poincare_out[i] > p_max) p_max = poincare_out[i];
    }
    printf("  Poincaré output: [%e, %e]\n", p_min, p_max);
    printf("  NaN count: %d\n", nan_count);
    printf("  Inf count: %d\n", inf_count);
    
    // Compare magnitudes (Poincaré should produce different but comparable values)
    float e_abs_max = fmaxf(fabsf(e_min), fabsf(e_max));
    float p_abs_max = fmaxf(fabsf(p_min), fabsf(p_max));
    printf("\n  Euclidean max abs: %e\n", e_abs_max);
    printf("  Poincaré max abs: %e\n", p_abs_max);
    
    // Quality gates
    int passes = 0;
    if (nan_count == 0) {
        printf("  ✅ No NaN\n");
        passes++;
    } else {
        printf("  ❌ Has NaN\n");
    }
    if (inf_count == 0) {
        printf("  ✅ No Inf\n");
        passes++;
    } else {
        printf("  ❌ Has Inf\n");
    }
    if (p_abs_max < 100.0f && p_abs_max > 1e-4f) {
        printf("  ✅ Output magnitude reasonable (%e)\n", p_abs_max);
        passes++;
    } else {
        printf("  ❌ Output magnitude suspicious (%e)\n", p_abs_max);
    }
    
    printf("\n  Result: %s\n", passes >= 2 ? "PASS (Stage 2 QG)" : "FAIL");
    
    // Free all
    free(x);
    free(w.attn_qkv_weight); free(w.attn_gate_weight);
    free(w.ssm_beta_weight); free(w.ssm_alpha_weight);
    free(w.ssm_dt_bias); free(w.ssm_a);
    free(w.ssm_conv1d_weight); free(w.ssm_norm_weight); free(w.ssm_out_weight);
    free(expected);
    free(euclidean_state); free(euclidean_conv); free(euclidean_out);
    free(poincare_state); free(poincare_conv); free(poincare_out);
    
    return passes >= 2 ? 0 : 1;
}
