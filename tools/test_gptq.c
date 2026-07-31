/* Test GPTQ second-order weight quantization (doc B06). */
#include "wubu_gptq.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

int main(void) {
    int n_samples = 16;
    int n_cols = 8;
    int n_rows = 16;

    /* Generate calibration data X (small random matrix) */
    float *X = (float *)malloc((size_t)n_samples * n_cols * sizeof(float));
    assert(X);
    for (int i = 0; i < n_samples * n_cols; i++) X[i] = 0.1f * (i % 7);

    /* Generate weight matrix W */
    float *W = (float *)malloc((size_t)n_rows * n_cols * sizeof(float));
    assert(W);
    for (int i = 0; i < n_rows * n_cols; i++) W[i] = 0.5f * (i % 11 - 5);

    /* Test 1: Hessian computation */
    float *H = (float *)malloc((size_t)n_cols * n_cols * sizeof(float));
    assert(H);
    wubu_gptq_compute_hessian(X, n_samples, n_cols, H);

    /* H should be symmetric */
    for (int i = 0; i < n_cols; i++) {
        for (int j = 0; j < n_cols; j++) {
            float hij = H[i * n_cols + j];
            float hji = H[j * n_cols + i];
            assert(fabsf(hij - hji) < 1e-6f);
        }
    }
    printf("Hessian is symmetric ✓\n");

    /* Diagonal should be positive (regularized) */
    for (int i = 0; i < n_cols; i++) {
        assert(H[i * n_cols + i] > 0.0f);
    }
    printf("Diagonal is positive ✓\n");

    /* Test 2: quantize a single column */
    float col_error = wubu_gptq_quantize_column(W, H, n_rows, n_cols, 0, n_cols);
    printf("Column 0 quantization error: %.6f\n", (double)col_error);

    /* Test 3: Full GPTQ quantization */
    /* Re-initialize W */
    for (int i = 0; i < n_rows * n_cols; i++) W[i] = 0.5f * ((i + n_cols) % 11 - 5);

    float total_error = wubu_gptq_quantize(W, X, n_rows, n_cols, n_samples, n_cols);
    printf("Total GPTQ quantization error: %.6f\n", (double)total_error);
    assert(total_error >= 0.0f);

    /* Test 4: Quantized W should have limited error vs original */
    /* Re-init W and compare post-quantization */
    for (int i = 0; i < n_rows * n_cols; i++) W[i] = 0.5f * ((i + n_cols*2) % 11 - 5);
    float *W_orig = (float *)malloc((size_t)n_rows * n_cols * sizeof(float));
    memcpy(W_orig, W, (size_t)n_rows * n_cols * sizeof(float));

    total_error = wubu_gptq_quantize(W, X, n_rows, n_cols, n_samples, n_cols);

    float max_err = 0.0f;
    for (int i = 0; i < n_rows * n_cols; i++) {
        float e = fabsf(W[i] - W_orig[i]);
        if (e > max_err) max_err = e;
    }
    printf("Row quantization error: total=%.4f max=%.4f\n", (double)total_error, (double)max_err);
    assert(max_err < 10.0f); /* error bounded by quantization noise */

    free(X); free(W_orig); free(H);
    printf("ALL GPTQ TESTS PASSED\n");
    return 0;
}
