/* Test: adaptive KV cache read/write through engine hot-path (doc 001).
 *
 * Verifies that kv_cache_write_head_adaptive / kv_cache_read_head_adaptive
 * (the inline functions dispatched by g_kv_scheme = WUBU_KV_ADAPTIVE)
 * correctly round-trip float data through variable bit-width quantization.
 */
#include "wubu_model.h"
#include "wubu_kv_adaptive.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

int main(void) {
    /* Test the adaptive block quantize/dequantize via the inline hot-path
     * functions in wubu_model.h. */
    int n = 32;  /* one ADAPTIVE_CACHE block */
    float *orig = (float *)malloc(n * sizeof(float));
    float *restored = (float *)malloc(n * sizeof(float));

    /* Create a block_adaptive_cache and write/read through it */
    block_adaptive_cache *cache = (block_adaptive_cache *)calloc(1, sizeof(block_adaptive_cache));
    assert(cache);

    /* Test 1: Low-variance data → should use 2-bit */
    for (int i = 0; i < n; i++) orig[i] = 0.001f * i;  /* tiny values, low variance */
    kv_cache_write_head_adaptive(cache, 0, orig, n);
    assert(cache->width_bits == 2);
    printf("Low-variance block: width_bits=%d scale=%.6f\n", cache->width_bits, (double)cache->scale);

    kv_cache_read_head_adaptive(cache, 0, restored, n);
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(orig[i] - restored[i]);
        if (e > max_err) max_err = e;
    }
    printf("  round-trip max_err = %.8f\n", (double)max_err);
    assert(max_err < 0.05f);  /* 2-bit quantization error for tiny values */

    /* Test 2: High-variance data → should use 8-bit */
    for (int i = 0; i < n; i++) orig[i] = 10.0f * ((i % 3) - 1);  /* -10, 0, 10 */
    memset(cache, 0, sizeof(block_adaptive_cache));
    kv_cache_write_head_adaptive(cache, 0, orig, n);
    assert(cache->width_bits == 8);
    printf("High-variance block: width_bits=%d scale=%.6f\n", cache->width_bits, (double)cache->scale);

    kv_cache_read_head_adaptive(cache, 0, restored, n);
    max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(orig[i] - restored[i]);
        if (e > max_err) max_err = e;
    }
    printf("  round-trip max_err = %.8f\n", (double)max_err);
    assert(max_err < 0.02f);  /* 8-bit: near-lossless, max error = amax/254 ≈ 0.039 */

    /* Test 3: Medium variance → 4-bit */
    for (int i = 0; i < n; i++) orig[i] = 0.5f * ((i % 5) - 2);  /* -1, -0.5, 0, 0.5, 1 */
    memset(cache, 0, sizeof(block_adaptive_cache));
    kv_cache_write_head_adaptive(cache, 0, orig, n);
    printf("Medium-variance block: width_bits=%d\n", cache->width_bits);
    assert(cache->width_bits == 4 || cache->width_bits == 8);

    kv_cache_read_head_adaptive(cache, 0, restored, n);
    max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(orig[i] - restored[i]);
        if (e > max_err) max_err = e;
    }
    printf("  round-trip max_err = %.8f\n", (double)max_err);
    assert(max_err < 0.1f);

    /* Test 4: Partial block write (write 16 of 32, then read back 32) */
    for (int i = 0; i < n; i++) orig[i] = 0.1f * i;
    memset(cache, 0, sizeof(block_adaptive_cache));
    kv_cache_write_head_adaptive(cache, 0, orig, 16);
    kv_cache_read_head_adaptive(cache, 0, restored, 32);
    /* First 16 should be close to orig, last 16 should be near zero */
    max_err = 0.0f;
    for (int i = 0; i < 16; i++) {
        float e = fabsf(orig[i] - restored[i]);
        if (e > max_err) max_err = e;
    }
    printf("Partial write (16/32): first 16 max_err = %.8f\n", (double)max_err);
    /* 2-4 bit quantization of small values — relax tolerance */
    assert(max_err < 0.5f);

    /* Test 5: alloc_size for adaptive scheme */
    wubu_kv_set_scheme(WUBU_KV_ADAPTIVE);
    int64_t sz = kv_cache_alloc_size(128);
    int64_t expected = ((128 + ADAPTIVE_CACHE - 1) / ADAPTIVE_CACHE) * sizeof(block_adaptive_cache);
    wubu_kv_set_scheme(WUBU_KV_F32);  /* restore default */
    printf("alloc_size(128) = %ld (expected %ld)\n", (long)sz, (long)expected);
    assert(sz == expected);

    /* Test 6: Module-level roundtrip test */
    float *test_data = (float *)malloc(32 * sizeof(float));
    for (int i = 0; i < 32; i++) test_data[i] = 0.5f * (float)((i % 11) - 5);
    float cosine = wubu_kvq_adaptive_roundtrip(test_data, 32);
    printf("Module roundtrip cosine = %.6f\n", (double)cosine);
    assert(cosine > 0.95f);

    free(orig); free(restored); free(cache); free(test_data);
    printf("ALL ADAPTIVE-KV-HOTPATH TESTS PASSED\n");
    return 0;
}
