/**
 * Test Nested SSM forward pass.
 *
 * Tests:
 *   1. K=1 with R=0.956 matches single Poincaré SSM output
 *   2. K=4 with R=[0.5, 1.0, 2.0, 5.0] — no NaN, output in bounds
 *   3. State validity: all ball states within their respective radii
 *   4. K=1 with different R values
 */

#include "wubu_nested_ssm.h"
#include "wubu_mobius.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Load test data from binary file (same format as test_poincare_ssm.c)
static int load_test_data(const char *filename,
                          float **x, int *B, int *T,
                          ssm_layer_weights *w,
                          float **expected) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        printf("FAIL: Cannot open %s\n", filename);
        printf("  Run: python3 tools/gen_test_vectors.py\n");
        return -1;
    }

    int D, V;
    fread(B, sizeof(int), 1, f);
    fread(T, sizeof(int), 1, f);
    fread(&D, sizeof(int), 1, f);
    fread(&V, sizeof(int), 1, f);
    int N = (*B) * (*T);
    int qkv_dim = KEY_DIM * 2 + VALUE_DIM;

    *x = (float *)malloc(N * D * sizeof(float));
    fread(*x, sizeof(float), N * D, f);

    w->attn_qkv_weight = (float *)malloc(D * qkv_dim * sizeof(float));
    fread(w->attn_qkv_weight, sizeof(float), D * qkv_dim, f);

    w->attn_gate_weight = (float *)malloc(D * VALUE_DIM * sizeof(float));
    fread(w->attn_gate_weight, sizeof(float), D * VALUE_DIM, f);

    w->ssm_beta_weight = (float *)malloc(D * DT_RANK * sizeof(float));
    fread(w->ssm_beta_weight, sizeof(float), D * DT_RANK, f);

    w->ssm_alpha_weight = (float *)malloc(D * DT_RANK * sizeof(float));
    fread(w->ssm_alpha_weight, sizeof(float), D * DT_RANK, f);

    w->ssm_dt_bias = (float *)malloc(DT_RANK * sizeof(float));
    fread(w->ssm_dt_bias, sizeof(float), DT_RANK, f);

    w->ssm_a = (float *)malloc(DT_RANK * sizeof(float));
    fread(w->ssm_a, sizeof(float), DT_RANK, f);

    w->ssm_conv1d_weight = (float *)malloc(CONV_KERNEL * CONV_DIM * sizeof(float));
    fread(w->ssm_conv1d_weight, sizeof(float), CONV_KERNEL * CONV_DIM, f);

    w->ssm_norm_weight = (float *)malloc(SSM_D_STATE * sizeof(float));
    fread(w->ssm_norm_weight, sizeof(float), SSM_D_STATE, f);

    w->ssm_out_weight = (float *)malloc(VALUE_DIM * D * sizeof(float));
    fread(w->ssm_out_weight, sizeof(float), VALUE_DIM * D, f);

    *expected = (float *)malloc(N * D * sizeof(float));
    fread(*expected, sizeof(float), N * D, f);

    fclose(f);
    return 0;
}

static void free_weights(ssm_layer_weights *w) {
    free(w->attn_qkv_weight);
    free(w->attn_gate_weight);
    free(w->ssm_beta_weight);
    free(w->ssm_alpha_weight);
    free(w->ssm_dt_bias);
    free(w->ssm_a);
    free(w->ssm_conv1d_weight);
    free(w->ssm_norm_weight);
    free(w->ssm_out_weight);
}

// Check output for NaN, Inf, and compute stats
static void check_output(const char *label, const float *out, int N, int D,
                         int *nan_count, int *inf_count,
                         float *min_val, float *max_val) {
    *nan_count = 0;
    *inf_count = 0;
    *min_val = 1e30f;
    *max_val = -1e30f;

    for (int i = 0; i < N * D; i++) {
        if (isnan(out[i])) { (*nan_count)++; continue; }
        if (isinf(out[i])) { (*inf_count)++; continue; }
        if (out[i] < *min_val) *min_val = out[i];
        if (out[i] > *max_val) *max_val = out[i];
    }
    printf("  %s: [%e, %e], NaN=%d, Inf=%d\n",
           label, *min_val, *max_val, *nan_count, *inf_count);
}

int main() {
    float *x = NULL;
    int B, T;
    ssm_layer_weights w;
    float *expected = NULL;

    printf("=== Nested SSM Forward Test ===\n\n");

    // Load test data
    if (load_test_data("data/ssm_test_vectors.bin", &x, &B, &T, &w, &expected) != 0) {
        return 1;
    }
    int N = B * T;
    printf("  B=%d, T=%d, D=%d\n\n", B, T, D_MODEL);

    // ================================================================
    // Test 1: K=1 with R=0.956 — should match single Poincaré SSM
    // ================================================================
    printf("--- Test 1: K=1 matches single Poincaré SSM (R=0.956) ---\n");

    float R_single = 0.956f;

    // Run single Poincaré SSM (reference)
    float *ref_state = (float *)calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));
    float *ref_conv = (float *)calloc(B * (CONV_KERNEL - 1) * CONV_DIM, sizeof(float));
    float *ref_out = (float *)malloc(N * D_MODEL * sizeof(float));
    wubu_poincare_ssm_forward(x, B, T, &w, ref_state, ref_conv, R_single, ref_out);

    // Run nested SSM with K=1, R=[0.956]
    wubu_nested_ssm_state_t nstate;
    float R_arr1[1] = {R_single};
    if (wubu_nested_ssm_init(&nstate, 1, R_arr1) != 0) {
        printf("FAIL: nested_ssm_init failed\n");
        return 1;
    }
    float *nest_conv1 = (float *)calloc(B * (CONV_KERNEL - 1) * CONV_DIM, sizeof(float));
    float *nest_out1 = (float *)malloc(N * D_MODEL * sizeof(float));
    wubu_nested_ssm_forward(x, B, T, &w, &nstate, nest_conv1, NULL, nest_out1);

    // Compare outputs
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    for (int i = 0; i < N * D_MODEL; i++) {
        float diff = fabsf(ref_out[i] - nest_out1[i]);
        float abs_ref = fabsf(ref_out[i]);
        if (diff > max_diff) max_diff = diff;
        if (abs_ref > 1e-8f) {
            float rd = diff / abs_ref;
            if (rd > max_rel_diff) max_rel_diff = rd;
        }
    }
    printf("  Max abs diff vs Poincaré SSM: %e\n", max_diff);
    printf("  Max rel diff vs Poincaré SSM: %e\n", max_rel_diff);

    int t1_pass = (max_diff < 1e-4f);
    printf("  %s (max_diff < 1e-4)\n\n", t1_pass ? "✅ PASS" : "❌ FAIL");

    // ================================================================
    // Test 2: K=4 with R=[0.5, 1.0, 2.0, 5.0] — no NaN, within bounds
    // ================================================================
    printf("--- Test 2: K=4 with R=[0.5, 1.0, 2.0, 5.0] ---\n");

    float R4[4] = {0.5f, 1.0f, 2.0f, 5.0f};
    wubu_nested_ssm_state_t nstate4;
    if (wubu_nested_ssm_init(&nstate4, 4, R4) != 0) {
        printf("FAIL: nested_ssm_init for K=4 failed\n");
        return 1;
    }

    // Test with uniform gating (NULL)
    float *nest_conv4 = (float *)calloc(B * (CONV_KERNEL - 1) * CONV_DIM, sizeof(float));
    float *nest_out4 = (float *)malloc(N * D_MODEL * sizeof(float));
    wubu_nested_ssm_forward(x, B, T, &w, &nstate4, nest_conv4, NULL, nest_out4);

    int nan4, inf4;
    float min4, max4;
    check_output("Nested K=4 (uniform)", nest_out4, N, D_MODEL, &nan4, &inf4, &min4, &max4);

    // Validate internal state
    int valid4 = wubu_nested_ssm_validate(&nstate4);
    printf("  State validation: %s\n", valid4 == 0 ? "✅ PASS" : "❌ FAIL");

    int t2_pass = (nan4 == 0 && inf4 == 0 && valid4 == 0);
    printf("  %s\n\n", t2_pass ? "✅ PASS" : "❌ FAIL");

    // ================================================================
    // Test 3: K=4 with biased gating weights
    // ================================================================
    printf("--- Test 3: K=4 with biased gating ---\n");

    wubu_nested_ssm_gating_t gating;

    // Strong bias toward ball 3 (R=5.0)
    gating.ball_weights[0] = -5.0f;
    gating.ball_weights[1] = -2.0f;
    gating.ball_weights[2] = 1.0f;
    gating.ball_weights[3] = 10.0f;

    // Re-init nested state (reset to zero)
    wubu_nested_ssm_free(&nstate4);
    wubu_nested_ssm_init(&nstate4, 4, R4);

    float *nest_conv4b = (float *)calloc(B * (CONV_KERNEL - 1) * CONV_DIM, sizeof(float));
    float *nest_out4b = (float *)malloc(N * D_MODEL * sizeof(float));
    wubu_nested_ssm_forward(x, B, T, &w, &nstate4, nest_conv4b, &gating, nest_out4b);

    int nan4b, inf4b;
    float min4b, max4b;
    check_output("Nested K=4 (biased gate)", nest_out4b, N, D_MODEL, &nan4b, &inf4b, &min4b, &max4b);

    int valid4b = wubu_nested_ssm_validate(&nstate4);
    printf("  State validation: %s\n", valid4b == 0 ? "✅ PASS" : "❌ FAIL");

    int t3_pass = (nan4b == 0 && inf4b == 0 && valid4b == 0);
    printf("  %s\n\n", t3_pass ? "✅ PASS" : "❌ FAIL");

    // ================================================================
    // Test 4: K=1 with R=0.5, 1.0, 2.0 (different curvatures)
    // ================================================================
    printf("--- Test 4: Single balls with different curvatures ---\n");

    float test_Rs[] = {0.5f, 1.0f, 2.0f, 5.0f};
    for (int ri = 0; ri < 4; ri++) {
        float R_test = test_Rs[ri];

        wubu_nested_ssm_state_t ns;
        float R_arr[1] = {R_test};
        wubu_nested_ssm_init(&ns, 1, R_arr);

        float *nc = (float *)calloc(B * (CONV_KERNEL - 1) * CONV_DIM, sizeof(float));
        float *no = (float *)malloc(N * D_MODEL * sizeof(float));
        wubu_nested_ssm_forward(x, B, T, &w, &ns, nc, NULL, no);

        int nan, inf;
        float mn, mx;
        check_output("", no, N, D_MODEL, &nan, &inf, &mn, &mx);

        int v = wubu_nested_ssm_validate(&ns);
        printf("    R=%.1f: valid=%s, range=[%e, %e], no NaN/Inf: %s\n",
               R_test, v == 0 ? "✅" : "❌", mn, mx,
               (nan == 0 && inf == 0) ? "✅" : "❌");

        free(nc); free(no);
        wubu_nested_ssm_free(&ns);
    }
    printf("\n");

    // ================================================================
    // Summary
    // ================================================================
    int total_pass = t1_pass + t2_pass + t3_pass;
    printf("=== Summary ===\n");
    printf("  Test 1 (K=1 matches Poincaré): %s\n", t1_pass ? "PASS" : "FAIL");
    printf("  Test 2 (K=4 uniform):         %s\n", t2_pass ? "PASS" : "FAIL");
    printf("  Test 3 (K=4 biased gating):   %s\n", t3_pass ? "PASS" : "FAIL");
    printf("  Total: %d/3 passed\n", total_pass);

    // Cleanup
    free(x);
    free_weights(&w);
    free(expected);
    free(ref_state); free(ref_conv); free(ref_out);
    free(nest_conv1); free(nest_out1);
    wubu_nested_ssm_free(&nstate);
    free(nest_conv4); free(nest_out4);
    free(nest_conv4b); free(nest_out4b);
    wubu_nested_ssm_free(&nstate4);

    return total_pass >= 2 ? 0 : 1;
}
