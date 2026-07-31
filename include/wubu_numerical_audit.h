/*
 * wubu_numerical_audit.h -- Numerical-stability audit of dequant paths (doc F03).
 *
 * Systematically tests dequantization paths for NaN/Inf, catastrophic
 * cancellation, overflow safety, and round-trip accuracy.
 *
 * Self-contained C11, no third-party deps.
 */

#ifndef WUBU_NUMERICAL_AUDIT_H
#define WUBU_NUMERICAL_AUDIT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_AUDIT_OK         0
#define WUBU_AUDIT_NAN        1
#define WUBU_AUDIT_INF        2
#define WUBU_AUDIT_LARGE_ERR  3
#define WUBU_AUDIT_LOW_COSINE 4
#define WUBU_AUDIT_ERR       -1

/* Function pointer types for quantize/dequantize pairs. */
typedef int (*wubu_quant_fn)(const float *z, uint8_t *out, int *width_bits,
                               float *out_scale, int n);
typedef int (*wubu_dequant_fn)(const uint8_t *packed, int width_bits,
                                float scale, float *out, int n);

/* Check if a float array is clean (no NaN, no Inf). */
int wubu_audit_check_clean(const float *data, int n);

/* Compute max relative/absolute error and cosine similarity. */
float wubu_audit_rel_error(const float *a, const float *b, int n);
float wubu_audit_abs_error(const float *a, const float *b, int n);
float wubu_audit_cosine(const float *a, const float *b, int n);

/* Audit a quantize→dequantize round-trip. */
int wubu_audit_roundtrip(const float *input, int n,
                          wubu_quant_fn quantize, wubu_dequant_fn dequantize,
                          float tolerance, float *out_error);

/* Run full audit on 7 extreme-value test cases. */
int wubu_audit_extreme_values(wubu_quant_fn quantize, wubu_dequant_fn dequantize,
                                float *results, int max_results);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_NUMERICAL_AUDIT_H */
