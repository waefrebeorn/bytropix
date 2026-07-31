/* Test: RoPE-aware KV prefetch (doc A10). */
#include "wubu_rope_prefetch.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

int main(void) {
    int block_size = 16, head_dim = 128, n_kv_heads = 8, n_blocks = 64;
    wubu_kv_cacheline_t *store = wubu_kv_cacheline_create(block_size, head_dim, n_kv_heads, n_blocks);
    assert(store);
    int block_ids[64];
    for (int i = 0; i < 64; i++) block_ids[i] = i;

    wubu_rope_prefetch_kv(store, block_ids, 64, 512, 2, 2);
    printf("Prefetch at pos=512 (block=32), lookback=2, lookahead=2\n");

    wubu_rope_prefetch_kv(store, block_ids, 64, 0, 2, 1);
    printf("Prefetch at pos=0 (boundary)\n");

    wubu_rope_prefetch_kv(store, block_ids, 64, 63 * 16, 1, 2);
    printf("Prefetch at last block (boundary)\n");

    float theta0 = wubu_rope_theta(0, 0, 128);
    float theta1 = wubu_rope_theta(0, 1, 128);
    assert(theta0 == 0.0f);
    assert(theta1 > 0.0f);
    printf("RoPE theta(0,0)=%.6f, theta(0,1)=%.6f\n", (double)theta0, (double)theta1);

    float t10 = wubu_rope_theta(0, 10, 128);
    float t20 = wubu_rope_theta(0, 20, 128);
    assert(t20 > t10);
    printf("RoPE theta monotonic in position (%.4f < %.4f)\n", (double)t10, (double)t20);

    wubu_kv_cacheline_free(store);
    printf("ALL ROPE-PREFETCH TESTS PASSED\n");
    return 0;
}
