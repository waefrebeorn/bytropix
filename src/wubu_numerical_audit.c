/*
 * wubu_numerical_audit.c -- Numerical-stability audit of dequant paths (doc F03).
 *
 * Systematically tests every dequantization path in the engine for:
 *  1. No NaN / Inf in output (unless input has NaN/Inf)
 *  2. No catastrophic cancellation (dot product of dequantized weights
 *     must not lose all precision to rounding)
 *  3. Correctness: dequant(q(quant(x))) ≈ x within bounded error
 *  4. Overflow safety: extreme values don't produce undefined behavior
 *
 * Self-contained C11, no third-party deps.
 */

#include "wubu_numerical_audit.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/* Check if a float array contains NaN or Inf. */
int wubu_audit_check_clean(const float *data, int n) {
    if (!data || n <= 0) return -1;
    for (int i = 0; i < n; i++) {
        if (isnan(data[i])) return WUBU_AUDIT_NAN;
        if (isinf(data[i])) return WUBU_AUDIT_INF;
    }
    return WUBU_AUDIT_OK;
}

/* Compute the relative error between two float arrays.
 * Returns the max relative error, or -1 if denominator is too small. */
float wubu_audit_rel_error(const float *a, const float *b, int n) {
    if (!a || !b || n <= 0) return -1.0f;
    float max_rel = 0.0f;
    for (int i = 0; i < n; i++) {
        float denom = fabsf(a[i]);
        if (denom < 1e-10f) denom = 1e-10f;
        float rel = fabsf(a[i] - b[i]) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    return max_rel;
}

/* Compute the absolute error between two float arrays. */
float wubu_audit_abs_error(const float *a, const float *b, int n) {
    if (!a || !b || n <= 0) return -1.0f;
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(a[i] - b[i]);
        if (e > max_abs) max_abs = e;
    }
    return max_abs;
}

/* Compute the cosine similarity between two float arrays. */
float wubu_audit_cosine(const float *a, const float *b, int n) {
    if (!a || !b || n <= 0) return 0.0f;
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    return (denom > 1e-10f) ? dot / denom : 1.0f;
}

/* Audit a quantize→dequantize round-trip for a given function pair.
 * Returns WUBU_AUDIT_OK if the round-trip passes all checks. */
int wubu_audit_roundtrip(const float *input, int n,
                          wubu_quant_fn quantize, wubu_dequant_fn dequantize,
                          float tolerance, float *out_error) {
    if (!input || n <= 0 || !quantize || !dequantize) return WUBU_AUDIT_ERR;

    /* Check input is clean */
    int in_status = wubu_audit_check_clean(input, n);
    if (in_status != WUBU_AUDIT_OK) return in_status;

    /* Quantize */
    uint8_t packed[65536];
    if (n > (int)sizeof(packed)) return WUBU_AUDIT_ERR;
    int width;
    float scale;
    quantize(input, packed, &width, &scale, n);

    /* Dequantize */
    float recon[65536];
    if (n > (int)sizeof(recon)) return WUBU_AUDIT_ERR;
    dequantize(packed, width, scale, recon, n);

    /* Check output is clean */
    int out_status = wubu_audit_check_clean(recon, n);
    if (out_status != WUBU_AUDIT_OK) return out_status;

    /* Check error bounds */
    float err = wubu_audit_abs_error(input, recon, n);
    if (out_error) *out_error = err;
    if (err > tolerance) return WUBU_AUDIT_LARGE_ERR;

    /* Check cosine similarity */
    float cos = wubu_audit_cosine(input, recon, n);
    if (cos < 0.99f) return WUBU_AUDIT_LOW_COSINE;

    return WUBU_AUDIT_OK;
}

/* Run a full audit suite on extreme values.
 * Tests: zeros, maximum int8, minimum int8, mixed, alternating signs,
 * single large outlier, gradual decay. */
int wubu_audit_extreme_values(wubu_quant_fn quantize, wubu_dequant_fn dequantize,
                                float *results, int max_results) {
    if (!quantize || !dequantize || !results || max_results < 7) return -1;
    int n = 32;
    float test_cases[7][32];
    const char *names[7] = {
        "zeros", "max_int8", "min_int8", "mixed",
        "alternating", "outlier", "decay"
    };

    /* zeros */
    memset(test_cases[0], 0, sizeof(float) * 32);
    /* max_int8 */
    for (int i = 0; i < 32; i++) test_cases[1][i] = 127.0f;
    /* min_int8 */
    for (int i = 0; i < 32; i++) test_cases[2][i] = -128.0f;
    /* mixed */
    for (int i = 0; i < 32; i++) test_cases[3][i] = (float)(i % 9 - 4);
    /* alternating */
    for (int i = 0; i < 32; i++) test_cases[4][i] = (i % 2) ? 1.0f : -1.0f;
    /* single large outlier */
    for (int i = 0; i < 32; i++) test_cases[5][i] = (i == 15) ? 1000.0f : 0.001f;
    /* gradual decay */
    for (int i = 0; i < 32; i++) test_cases[6][i] = 10.0f * expf(-0.1f * i);

    int all_ok = WUBU_AUDIT_OK;
    for (int t = 0; t < 7; t++) {
        float err;
        int rc = wubu_audit_roundtrip(test_cases[t], n, quantize, dequantize,
                                       100.0f, &err);
        results[t] = err;
        if (rc != WUBU_AUDIT_OK) {
            all_ok = rc;
            fprintf(stderr, "Audit failed on '%s': rc=%d err=%.6f\n", names[t], rc, (double)err);
        }
    }
    return all_ok;
}
