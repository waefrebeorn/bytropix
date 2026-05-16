#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ============================================================
// Test: Chunked vs Sequential DeltaNet SSM Recurrence
// ============================================================
//
// Generates random inputs, runs both sequential and chunked
// recurrence, and compares the output using cosine similarity.
// The chunked output must match sequential within 1e-5 rel error.

static float rand_float(float lo, float hi) {
    return lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}

static int has_nan_or_inf(const float *a, int n) {
    for (int i = 0; i < n; i++) {
        if (isnan(a[i]) || isinf(a[i])) return 1;
    }
    return 0;
}

static float compute_cos_sim(const float *a, const float *b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        if (isnan(a[i]) || isinf(a[i]) || isnan(b[i]) || isinf(b[i])) return -2.0f;
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    double denom = sqrt(na) * sqrt(nb);
    if (denom < 1e-30) return 1.0f;  // both zero
    return (float)(dot / denom);
}

static float compute_max_rel_error(const float *a, const float *b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        if (isnan(a[i]) || isinf(a[i]) || isnan(b[i]) || isinf(b[i])) return -1.0f;  // sentinel for NaN
        float abs_a = fabsf(a[i]);
        float abs_b = fabsf(b[i]);
        float denom = fmaxf(abs_a, abs_b);
        if (denom < 1e-12f) denom = 1.0f;  // both near zero
        float err = fabsf(a[i] - b[i]) / denom;
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static int test_chunked_vs_sequential(int T, int seed) {
    const int d = SSM_D_STATE;
    const int n_heads = SSM_V_HEADS;
    const int n_kheads = SSM_K_HEADS;
    const int B = 1;
    const int N = B * T;

    printf("  T=%d, seed=%d ... ", T, seed);
    fflush(stdout);

    srand(seed);

    // Allocate inputs
    float *q_norm = (float *)malloc(N * n_kheads * d * sizeof(float));
    float *k_norm = (float *)malloc(N * n_kheads * d * sizeof(float));
    float *v_conv = (float *)malloc(N * n_heads * d * sizeof(float));
    float *beta_flat = (float *)malloc(N * DT_RANK * sizeof(float));
    float *gate_flat = (float *)malloc(N * DT_RANK * sizeof(float));

    if (!q_norm || !k_norm || !v_conv || !beta_flat || !gate_flat) {
        printf("ALLOC FAIL\n");
        return 1;
    }

    // Generate random Q (L2-normalized already, but we just use raw random)
    // Sequential and chunked both use same input, so distribution doesn't matter
    for (int i = 0; i < N * n_kheads * d; i++) q_norm[i] = rand_float(-1.0f, 1.0f);
    for (int i = 0; i < N * n_kheads * d; i++) k_norm[i] = rand_float(-1.0f, 1.0f);
    for (int i = 0; i < N * n_heads * d; i++) v_conv[i] = rand_float(-1.0f, 1.0f);
    
    // Beta and gate: important these are in reasonable ranges
    // Beta: [0, 1] since it's sigmoid output
    // Gate: must be small to avoid exponential blowup with 500 tokens
    // Real model uses alpha_softplus * ssm_a where ssm_a ~ -log(something small)
    // Realistic range: [-0.5, 0.5] → exp(gate) ∈ [0.6, 1.65], stable for 500 tokens
    for (int i = 0; i < N * DT_RANK; i++) {
        beta_flat[i] = rand_float(0.0f, 1.0f);
        gate_flat[i] = rand_float(-0.5f, 0.5f);
    }

    // Allocate states and outputs
    int state_sz = n_heads * d * d;
    float *state_seq = (float *)calloc(state_sz, sizeof(float));
    float *state_chk = (float *)calloc(state_sz, sizeof(float));
    float *out_seq = (float *)calloc(N * n_heads * d, sizeof(float));
    float *out_chk = (float *)calloc(N * n_heads * d, sizeof(float));

    if (!state_seq || !state_chk || !out_seq || !out_chk) {
        printf("ALLOC FAIL\n");
        return 1;
    }

    // Initialize states identically
    // Use same random initial state for both
    for (int i = 0; i < state_sz; i++) {
        float v = rand_float(-0.1f, 0.1f);
        state_seq[i] = v;
        state_chk[i] = v;
    }

    // Run sequential
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    wubu_ssm_sequential_recurrence(B, T, q_norm, k_norm, v_conv,
                                    beta_flat, gate_flat, state_seq, out_seq);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double time_seq = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    // Run chunked
    clock_gettime(CLOCK_MONOTONIC, &t0);
    wubu_ssm_chunked_recurrence(B, T, q_norm, k_norm, v_conv,
                                 beta_flat, gate_flat, state_chk, out_chk);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double time_chk = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    // Compare outputs
    float cos_sim_out = compute_cos_sim(out_seq, out_chk, N * n_heads * d);
    float max_rel_err_out = compute_max_rel_error(out_seq, out_chk, N * n_heads * d);

    // Compare states
    float cos_sim_state = compute_cos_sim(state_seq, state_chk, state_sz);
    float max_rel_err_state = compute_max_rel_error(state_seq, state_chk, state_sz);

    printf("cos_sim_out=%.10f max_rel_err=%.2e cos_sim_state=%.10f "
           "time_seq=%.4fs time_chk=%.4fs speedup=%.2fx",
           cos_sim_out, max_rel_err_out, cos_sim_state,
           time_seq, time_chk, time_seq / (time_chk + 1e-12));

    int pass = 1;
    if (has_nan_or_inf(out_seq, N * n_heads * d) || has_nan_or_inf(out_chk, N * n_heads * d)) {
        printf(" NAN_INF");
        pass = 0;
    }
    if (cos_sim_out < 1.0f - 1e-10f) {
        printf(" FAIL_OUT_COS");
        pass = 0;
    }
    if (max_rel_err_out > 1e-5f && max_rel_err_out > 0.0f) {
        printf(" FAIL_OUT_ERR");
        pass = 0;
    }
    if (has_nan_or_inf(state_seq, state_sz) || has_nan_or_inf(state_chk, state_sz)) {
        printf(" STATE_NAN");
        pass = 0;
    }
    if (cos_sim_state < 1.0f - 1e-10f) {
        printf(" FAIL_STATE_COS");
        pass = 0;
    }
    if (max_rel_err_state > 1e-5f && max_rel_err_state > 0.0f) {
        printf(" FAIL_STATE_ERR");
        pass = 0;
    }

    if (pass) {
        printf(" PASS\n");
    } else {
        printf(" FAIL\n");
    }

    free(q_norm);
    free(k_norm);
    free(v_conv);
    free(beta_flat);
    free(gate_flat);
    free(state_seq);
    free(state_chk);
    free(out_seq);
    free(out_chk);

    return pass ? 0 : 1;
}

int main() {
    printf("=== Chunked vs Sequential DeltaNet SSM Recurrence ===\n\n");

    int n_fail = 0;

    // Test at CHUNK_SIZE boundaries and various sizes
    int test_sizes[] = {64, 65, 128, 192, 256, 512, 128, 64, 96, 160, 200, 300, 400, 500};
    int n_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int i = 0; i < n_tests; i++) {
        int T = test_sizes[i];
        int seed = 42 + i;
        n_fail += test_chunked_vs_sequential(T, seed);
    }

    printf("\n=== Results: %d/%d passed ===\n", n_tests - n_fail, n_tests);

    return n_fail > 0 ? 1 : 0;
}
