/*
 * wubu_gptq.c -- GPTQ second-order weight quantization (doc B06).
 *
 * Source: Frantar et al., "GPTQ: Accurate Post-Training Quantization for
 * Generative Pre-trained Transformers", ICLR 2023.
 *
 * Core idea: Quantize weights column-by-column using second-order information
 * (Hessian) from calibration data. For each column i, the quantization error
 * is compensated by updating the remaining unquantized columns:
 *
 *   W[:, i] -= (quant_error_i / H[i,i]) * H[:, i]
 *
 * This "pushes" the error from quantizing column i into the still-unquantized
 * columns, minimizing the overall output error.
 *
 * The Hessian is approximated as H = X^T X where X is the calibration input.
 * The Lazy-Batch-Apple (LBA) variant processes columns in batches for speed.
 *
 * Self-contained C11, no third-party deps. Offline calibration tool.
 */

#include "wubu_gptq.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Compute Hessian H = X^T X / n_samples, where X is [n_samples, n_features].
 * H is [n_features, n_features], symmetric positive semi-definite. */
void wubu_gptq_compute_hessian(const float *X, int n_samples, int n_features,
                                 float *H) {
    if (!X || !H || n_samples <= 0 || n_features <= 0) return;

    /* H[j, k] = sum_i X[i, j] * X[i, k] / n_samples */
    memset(H, 0, n_features * n_features * sizeof(float));
    for (int i = 0; i < n_samples; i++) {
        const float *xi = X + (size_t)i * n_features;
        for (int j = 0; j < n_features; j++) {
            for (int k = j; k < n_features; k++) {
                H[j * n_features + k] += xi[j] * xi[k];
            }
        }
    }
    /* Normalize and symmetrize */
    float inv_n = 1.0f / (float)n_samples;
    for (int j = 0; j < n_features; j++) {
        for (int k = j; k < n_features; k++) {
            H[j * n_features + k] *= inv_n;
            H[k * n_features + j] = H[j * n_features + k]; /* symmetric */
        }
        /* Add small diagonal for numerical stability */
        H[j * n_features + j] += 1e-6f;
    }
}

/* Quantize a single column using GPTQ error compensation.
 * W: [n_rows, n_cols] row-major weight matrix (modified in place).
 * H: [n_cols, n_cols] Hessian.
 * col: the column index to quantize.
 * group_size: number of columns per quantization group (e.g. 128).
 *
 * Returns the quantization error for this column. */
float wubu_gptq_quantize_column(float *W, const float *H, int n_rows, int n_cols,
                                 int col, int group_size) {
    if (!W || !H || col < 0 || col >= n_cols) return 0.0f;

    /* Quantize each row of column `col` to int8 with absmax scaling */
    float max_val = 0.0f;
    for (int r = 0; r < n_rows; r++) {
        float a = fabsf(W[r * n_cols + col]);
        if (a > max_val) max_val = a;
    }
    float scale = max_val / 127.0f;
    if (scale < 1e-10f) scale = 1e-10f;

    float total_error = 0.0f;
    for (int r = 0; r < n_rows; r++) {
        int idx = r * n_cols + col;
        float orig = W[idx];
        int q = (int)roundf(orig / scale);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        float dequant = (float)q * scale;
        float err = orig - dequant;
        total_error += fabsf(err);
        W[idx] = dequant;

        /* GPTQ error compensation: update remaining columns of this row.
         * W[r, col+1:] -= err / H[col, col] * H[col, col+1:] */
        float h_inv = 1.0f / H[col * n_cols + col];
        for (int c = col + 1; c < n_cols && c < col + group_size; c++) {
            W[r * n_cols + c] -= err * h_inv * H[col * n_cols + c];
        }
    }

    return total_error;
}

/* Full GPTQ quantization: quantize all columns in order, compensating error.
 * W: [n_rows, n_cols] — modified in place to dequantized values.
 * X: [n_samples, n_cols] — calibration data.
 * Returns the total quantization error. */
float wubu_gptq_quantize(float *W, const float *X, int n_rows, int n_cols,
                          int n_samples, int group_size) {
    if (!W || !X || n_rows <= 0 || n_cols <= 0) return -1.0f;
    if (group_size <= 0) group_size = n_cols; /* all in one group */

    /* Step 1: compute Hessian H = X^T X / n_samples */
    float *H = (float *)malloc((size_t)n_cols * n_cols * sizeof(float));
    if (!H) return -1.0f;

    wubu_gptq_compute_hessian(X, n_samples, n_cols, H);

    /* Step 2: quantize columns in order, compensating error */
    float total_error = 0.0f;
    for (int col = 0; col < n_cols; col++) {
        total_error += wubu_gptq_quantize_column(W, H, n_rows, n_cols, col, group_size);
    }

    free(H);
    return total_error;
}
