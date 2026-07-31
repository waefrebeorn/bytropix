/* Test: LMCache prefix+PD KV persistence (doc A06). */
#include "wubu_lmcache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <sys/stat.h>

int main(void) {
    /* Use /tmp for cache dir */
    const char *cache_dir = "/tmp/wubu_lmcache_test";
    mkdir(cache_dir, 0755);

    int n_layers = 2, block_size = 4, head_dim = 8, n_kv_heads = 2;
    int n_blocks = 3;

    wubu_lmcache_t *c = wubu_lmcache_create(cache_dir, n_layers, block_size, head_dim, n_kv_heads);
    assert(c);

    /* Build fake KV data and token sequence */
    int n_tokens = 12;
    int tokens[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    size_t kv_size = (size_t)n_layers * n_blocks * n_kv_heads * block_size * head_dim;
    float *kv_data = (float *)malloc(kv_size * sizeof(float));
    float *kv_out = (float *)malloc(kv_size * sizeof(float));
    for (size_t i = 0; i < kv_size; i++) kv_data[i] = 0.001f * i;

    /* Test 1: First request — miss, then store */
    int loaded = wubu_lmcache_load(c, "test-model", tokens, n_tokens, kv_out, n_blocks);
    printf("Request 1 (load): %d blocks (expected 0, first-time miss)\n", loaded);
    assert(loaded == 0);

    int stored = wubu_lmcache_store(c, "test-model", tokens, n_tokens, kv_data, n_blocks);
    printf("Request 1 (store): %d blocks stored\n", stored);
    assert(stored == n_blocks);

    /* Test 2: Second request with same prefix — hit! */
    loaded = wubu_lmcache_load(c, "test-model", tokens, n_tokens, kv_out, n_blocks);
    printf("Request 2 (load): %d blocks (expected %d, cache hit)\n", loaded, n_blocks);
    assert(loaded == n_blocks);

    /* Verify data integrity — for 2-bit, error is up to 1 LSB = scale */
    float max_err = 0.0f;
    for (int i = 0; i < kv_size; i++) {
        float e = fabsf(kv_data[i] - kv_out[i]);
        if (e > max_err) max_err = e;
    }
    printf("Data integrity: max_err = %.8f (2-bit LSB = %.8f)\n",
           (double)max_err, (double)0.01033333f);
    /* 2-bit quantization: error < 1.5 LSB is acceptable */
    assert(max_err < 0.02f);

    /* Test 3: Different model or tokens — miss */
    int tokens2[12] = {99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88};
    loaded = wubu_lmcache_load(c, "test-model", tokens2, n_tokens, kv_out, n_blocks);
    printf("Different tokens (load): %d blocks (expected 0, miss)\n", loaded);
    assert(loaded == 0);

    loaded = wubu_lmcache_load(c, "other-model", tokens, n_tokens, kv_out, n_blocks);
    printf("Different model (load): %d blocks (expected 0, miss)\n", loaded);
    assert(loaded == 0);

    /* Test 4: Stats */
    int n_entries; size_t hits, misses, evict;
    wubu_lmcache_stats(c, &n_entries, &hits, &misses, &evict);
    printf("Stats: entries=%d hits=%zu misses=%zu evictions=%zu\n",
           n_entries, hits, misses, evict);
    assert(hits == 1);
    assert(misses >= 2);
    float hr = wubu_lmcache_hit_rate(c);
    printf("Hit rate: %.2f%%\n", (double)(hr * 100.0f));
    assert(hr > 0.0f);

    free(kv_data); free(kv_out);
    wubu_lmcache_free(c);
    printf("ALL LMCACHE TESTS PASSED\n");
    return 0;
}
