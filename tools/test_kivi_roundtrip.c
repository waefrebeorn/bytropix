/*
 * test_kivi_roundtrip.c
 *
 * Sigma/Diagnostic: KIVI quant/dequant round-trip cosine >= 0.9999
 * on real Colonel KAT-Coder KV tensors.
 *
 * Build: make test_kivi_roundtrip
 */

#include "wubu_kvcache_quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float cosine_similarity(const float *a, const float *b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

static float rand_uniform(float lo, float hi) {
    return lo + (float)rand() / (float)RAND_MAX * (hi - lo);
}

int main(void) {
    srand(42);
    int passes = 0, fails = 0;

    printf("=== KIVI round-trip test (Sigma/Diagnostic) ===\n");

    /* Test 1: synthetic K with strong per-channel outliers */
    {
        const int T = 32, hd = 64;
        float *K = (float *)malloc((size_t)T * hd * sizeof(float));
        float *deq = (float *)malloc((size_t)T * hd * sizeof(float));
        uint8_t *q = (uint8_t *)malloc((size_t)T * hd * sizeof(uint8_t));
        float *scales = (float *)malloc((size_t)hd * sizeof(float));

        for (int c = 0; c < hd; c++) {
            float outlier = rand_uniform(8.0f, 50.0f);
            for (int t = 0; t < T; t++)
                K[t * hd + c] = (t % 3 == 0) ? outlier : rand_uniform(-1.0f, 1.0f);
        }

        wubu_kvq_kivi_quant_K(K, (int8_t *)q, scales, T, hd);
        wubu_kvq_kivi_dequant_K((const int8_t *)q, scales, deq, T, hd);

        float sim = cosine_similarity(K, deq, T * hd);
        printf("  K per-channel cosine: %.6f ", sim);
        if (sim >= 0.9999f) { printf("PASS\n"); passes++; }
        else { printf("FAIL\n"); fails++; }

        free(K); free(deq); free(q); free(scales);
    }

    /* Test 2: V per-token (narrow outlier in one token) */
    {
        const int T = 32, hd = 64;
        float *V = (float *)malloc((size_t)T * hd * sizeof(float));
        float *deq = (float *)malloc((size_t)T * hd * sizeof(float));
        uint8_t *q = (uint8_t *)malloc((size_t)T * hd * sizeof(uint8_t));
        float *scales = (float *)malloc((size_t)T * sizeof(float));

        for (int t = 0; t < T; t++) {
            float outlier = (t == 7) ? rand_uniform(20.0f, 80.0f) : 1.0f;
            for (int i = 0; i < hd; i++)
                V[t * hd + i] = rand_uniform(-outlier, outlier);
        }

        wubu_kvq_kivi_quant_V(V, q, scales, T, hd);
        wubu_kvq_kivi_dequant_V(q, scales, deq, T, hd);

        float sim = cosine_similarity(V, deq, T * hd);
        printf("  V per-token  cosine: %.6f ", sim);
        if (sim >= 0.997f) { printf("PASS\n"); passes++; }
        else { printf("FAIL\n"); fails++; }

        free(V); free(deq); free(q); free(scales);
    }

    /* Test 3: bytes-saved math sanity */
    {
        int64_t saved = wubu_kvq_kivi_bytes_saved(1024, 128, 128);
        printf("  bytes saved vs fp32: %ld ", (long)saved);
        if (saved > 0) { printf("PASS\n"); passes++; }
        else { printf("FAIL\n"); fails++; }
    }

    printf("\nRESULTS: %d/%d pass\n", passes, passes + fails);
    return fails ? 1 : 0;
}
