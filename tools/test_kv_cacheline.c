/* Test: cache-line-aligned KV page storage (doc C03/I03). */
#include "wubu_kv_cacheline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>

int main(void) {
    int block_size = 16, head_dim = 128, n_kv_heads = 8, n_blocks = 32;
    wubu_kv_cacheline_t *store = wubu_kv_cacheline_create(block_size, head_dim, n_kv_heads, n_blocks);
    assert(store);
    assert(((uintptr_t)store->K_data & 63) == 0);
    assert(((uintptr_t)store->V_data & 63) == 0);
    printf("K_data aligned to 64B\n");
    printf("V_data aligned to 64B\n");
    for (int b = 0; b < n_blocks; b++) assert(wubu_kv_cacheline_is_aligned(store, b));
    printf("All %d blocks cache-line aligned\n", n_blocks);
    int n_floats = n_kv_heads * head_dim;
    float *k_in = (float *)malloc(n_floats * sizeof(float));
    float *v_in = (float *)malloc(n_floats * sizeof(float));
    float *k_out = (float *)malloc(n_floats * sizeof(float));
    float *v_out = (float *)malloc(n_floats * sizeof(float));
    for (int i = 0; i < n_floats; i++) { k_in[i] = 0.1f * i; v_in[i] = -0.1f * i; }
    wubu_kv_cacheline_write(store, 5, 3, k_in, v_in);
    wubu_kv_cacheline_read(store, 5, 3, k_out, v_out);
    for (int i = 0; i < n_floats; i++) { assert(k_out[i] == k_in[i]); assert(v_out[i] == v_in[i]); }
    printf("Write+read round-trip correct\n");
    size_t total, per_block; int n_blk;
    wubu_kv_cacheline_stats(store, &total, &per_block, &n_blk);
    assert(n_blk == n_blocks); assert(total > 0); assert(per_block > 0);
    printf("Stats: total=%zu per_block=%zu n_blocks=%d\n", total, per_block, n_blk);
    free(k_in); free(v_in); free(k_out); free(v_out);
    wubu_kv_cacheline_free(store);
    printf("ALL KV-CACHELINE TESTS PASSED\n");
    return 0;
}
