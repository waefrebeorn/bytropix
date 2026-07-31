/* Test: FlashAttention-style fused prefill (doc H01).
 *
 * Verifies that the tiled online-softmax attention produces the same
 * result as the naive O(S²) reference implementation.
 */
#include "wubu_flash_prefill.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

/* Naive reference attention: out = softmax(Q·K^T / sqrt(d)) · V */
static void ref_attn(const float *Q, const float *K, const float *V,
                     float *out, int n_heads, int seq_len, int head_dim) {
    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);
    float *scores = (float *)malloc(seq_len * sizeof(float));
    float *out_h = (float *)malloc(head_dim * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        const float *Qh = Q + (size_t)h * seq_len * head_dim;
        const float *Kh = K + (size_t)h * seq_len * head_dim;
        const float *Vh = V + (size_t)h * seq_len * head_dim;
        float *Oh = out + (size_t)h * seq_len * head_dim;

        for (int i = 0; i < seq_len; i++) {
            /* Compute scores */
            float max_s = -1e30f;
            for (int j = 0; j < seq_len; j++) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    dot += Qh[i * head_dim + d] * Kh[j * head_dim + d];
                scores[j] = dot * inv_sqrt_d;
                if (scores[j] > max_s) max_s = scores[j];
            }
            /* Softmax */
            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                scores[j] = expf(scores[j] - max_s);
                sum_exp += scores[j];
            }
            /* Weighted V */
            memset(out_h, 0, head_dim * sizeof(float));
            for (int j = 0; j < seq_len; j++) {
                for (int d = 0; d < head_dim; d++)
                    out_h[d] += scores[j] * Vh[j * head_dim + d];
            }
            float inv = 1.0f / sum_exp;
            for (int d = 0; d < head_dim; d++)
                Oh[i * head_dim + d] = out_h[d] * inv;
        }
    }
    free(scores); free(out_h);
}

int main(void) {
    int n_heads = 2;
    int seq_len = 16;
    int head_dim = 8;

    size_t sz = (size_t)n_heads * seq_len * head_dim;
    float *Q = (float *)malloc(sz * sizeof(float));
    float *K = (float *)malloc(sz * sizeof(float));
    float *V = (float *)malloc(sz * sizeof(float));
    float *out_ref = (float *)calloc(sz, sizeof(float));
    float *out_flash = (float *)calloc(sz, sizeof(float));

    /* Initialize with small random values */
    for (size_t i = 0; i < sz; i++) {
        Q[i] = 0.1f * (float)((i * 7) % 11 - 5);
        K[i] = 0.1f * (float)((i * 13) % 11 - 5);
        V[i] = 0.1f * (float)((i * 17) % 11 - 5);
    }

    /* Compute reference */
    ref_attn(Q, K, V, out_ref, n_heads, seq_len, head_dim);

    /* Compute FlashAttention (tile=4 for small seq_len) */
    wubu_flash_prefill_attn(Q, K, V, out_flash, n_heads, seq_len, head_dim, 4);

    /* Compare: max absolute error */
    float max_err = 0.0f;
    for (size_t i = 0; i < sz; i++) {
        float e = fabsf(out_ref[i] - out_flash[i]);
        if (e > max_err) max_err = e;
    }
    printf("FlashAttention vs reference: max_err = %.8f (seq=%d, dim=%d, heads=%d, tile=4)\n",
           (double)max_err, seq_len, head_dim, n_heads);
    assert(max_err < 1e-4f);

    /* Test with larger sequence to verify tiling works */
    memset(out_ref, 0, sz * sizeof(float));
    memset(out_flash, 0, sz * sizeof(float));

    /* Re-init with fixed seed pattern */
    for (size_t i = 0; i < sz; i++) {
        Q[i] = 0.01f * (float)((i * 3 + 1) % 17 - 8);
        K[i] = 0.01f * (float)((i * 5 + 2) % 17 - 8);
        V[i] = 0.01f * (float)((i * 7 + 3) % 17 - 8);
    }
    ref_attn(Q, K, V, out_ref, n_heads, seq_len, head_dim);
    wubu_flash_prefill_attn(Q, K, V, out_flash, n_heads, seq_len, head_dim, 8);

    max_err = 0.0f;
    for (size_t i = 0; i < sz; i++) {
        float e = fabsf(out_ref[i] - out_flash[i]);
        if (e > max_err) max_err = e;
    }
    printf("FlashAttention vs reference (tile=8): max_err = %.8f\n", (double)max_err);
    assert(max_err < 1e-4f);

    /* Test with tile=seq_len (single tile = naive softmax) */
    memset(out_flash, 0, sz * sizeof(float));
    wubu_flash_prefill_attn(Q, K, V, out_flash, n_heads, seq_len, head_dim, seq_len);
    max_err = 0.0f;
    for (size_t i = 0; i < sz; i++) {
        float e = fabsf(out_ref[i] - out_flash[i]);
        if (e > max_err) max_err = e;
    }
    printf("FlashAttention vs reference (tile=seq): max_err = %.8f\n", (double)max_err);
    assert(max_err < 1e-5f);

    free(Q); free(K); free(V); free(out_ref); free(out_flash);
    printf("ALL FLASH-PREFILL TESTS PASSED\n");
    return 0;
}
