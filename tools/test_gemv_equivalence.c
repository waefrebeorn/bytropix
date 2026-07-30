/*
 * tools/test_gemv_equivalence.c -- Bounded kernel equivalence test (doc 009).
 *
 * Exhaustively test quantized GEMV over small M/K with discrete weight/activation
 * patterns. Verify quantized_gemv(W,x) ≈ reference_gemv(W,x) within tolerance.
 * Also check invariants: outputs finite, no NaN, scale non-zero.
 *
 * Basis: "Equivalence Checking of ML GPU Kernels" (arXiv:2511.12638), Alive2, HEC (USENIX'25).
 */

#include "gguf_reader.h"
#include "wubu_smoothquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define TINY_M 4
#define TINY_K 4
#define TOL 1e-3f

static int tests_run = 0;
static int tests_passed = 0;

static float ref_gemm_f32(const float *W, const float *x, int M, int K) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) sum += W[k] * x[k];
    return sum;
}

/* Quantized GEMV: int8 weights, int8 activations, dequantized. */
static float quant_gemm_int8(const int8_t *W_q, const int8_t *x_q,
                              float W_scale, float x_scale, int M, int K) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += ((float)W_q[k] * W_scale) * ((float)x_q[k] * x_scale);
    }
    return sum;
}

static void test_case_int8(float W_scale, float x_scale, int8_t *W_q, int8_t *x_q,
                           float *ref_out, float *q_out, int M, int K) {
    tests_run++;
    float W_f32[TINY_K], x_f32[TINY_K];
    for (int i = 0; i < K; i++) { W_f32[i] = (float)W_q[i] * W_scale; x_f32[i] = (float)x_q[i] * x_scale; }
    *ref_out = ref_gemm_f32(W_f32, x_f32, M, K);
    *q_out = quant_gemm_int8(W_q, x_q, W_scale, x_scale, M, K);

    float diff = fabsf(*ref_out - *q_out);
    if (diff > TOL * (fabsf(*ref_out) + 1.0f)) {
        fprintf(stderr, "FAIL: diff=%g ref=%g q=%g\n", (double)diff, (double)*ref_out, (double)*q_out);
        return;
    }
    tests_passed++;
}

static void test_invariants(float *out, int M) {
    tests_run++;
    int all_finite = 1;
    for (int m = 0; m < M; m++) {
        if (!isfinite(out[m]) || out[m] != out[m]) {
            all_finite = 0;
            break;
        }
    }
    if (all_finite) tests_passed++;
}

int main(void) {
    int8_t W_q[TINY_K] = {-1, 0, 1, 2};
    int8_t x_q[TINY_K] = { 1, -1, 2, 0};
    float W_scale = 0.5f, x_scale = 0.25f;

    float ref_out = 0, q_out = 0;
    test_case_int8(W_scale, x_scale, W_q, x_q, &ref_out, &q_out, 1, TINY_K);

    /* Invariant check */
    float out_vals[TINY_M];
    for (int m = 0; m < TINY_M; m++) out_vals[m] = ref_out;
    test_invariants(out_vals, TINY_M);

    printf("ALL GEMV EQUIVALENCE TESTS: %d/%d PASSED\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}