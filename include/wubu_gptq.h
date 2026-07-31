/*
 * wubu_gptq.h -- GPTQ second-order weight quantization (doc B06).
 *
 * Source: Frantar et al., "GPTQ", ICLR 2023.
 *
 * Quantizes weights column-by-column using second-order error
 * compensation from a Hessian of the calibration data.
 *
 * Self-contained C11, no third-party deps. Offline calibration tool.
 */

#ifndef WUBU_GPTQ_H
#define WUBU_GPTQ_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compute the Hessian H = X^T X / n_samples for calibration data X.
 * X:  [n_samples, n_features] row-major
 * H:  [n_features, n_features] output (symmetric, diagonal-regularized) */
void wubu_gptq_compute_hessian(const float *X, int n_samples, int n_features,
                                           float *H);

/* Quantize a single column using GPTQ error compensation.
 * W:        [n_rows, n_cols] row-major (modified in place)
 * H:        [n_cols, n_cols] Hessian
 * col:      column index to quantize
 * group_size: quantization group size (affects error compensation scope)
 * Returns total absolute quantization error for this column. */
float wubu_gptq_quantize_column(float *W, const float *H, int n_rows, int n_cols,
                                            int col, int group_size);

/* Full GPTQ quantization: quantize all columns, error-compensating each.
 * W:    [n_rows, n_cols] — weight matrix (modified in place)
 * X:    [n_samples, n_cols] — calibration input data
 * Returns total quantization error. */
float wubu_gptq_quantize(float *W, const float *X, int n_rows, int n_cols,
                                int n_samples, int group_size);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_GPTQ_H */
