/*
 * wubu_rope_prefetch.c -- RoPE-aware KV cache prefetch (doc A10).
 *
 * Source: Meta Llama-2 inference optimizations; RoPE (Su et al., 2021)
 * position encoding means K vectors at nearby positions have similar
 * rotation angles. During decode at position pos, the attention weights
 * concentrate on nearby tokens (locality bias), so prefetching the KV
 * blocks for positions [pos-k, pos+k] before computing attention reduces
 * cache miss latency.
 *
 * On a CPU engine, this is a software prefetch hint: __builtin_prefetch
 * on the KV block data for the next few positions, overlapped with the
 * current position's compute.
 *
 * Self-contained C11, no third-party deps.
 */

#include "wubu_rope_prefetch.h"
#include <string.h>

/* Prefetch KV blocks for nearby positions (software prefetch).
 * store:     cache-line aligned KV store
 * block_ids: array of physical block IDs for the current sequence
 * n_blocks:  number of blocks in the sequence
 * pos:       current decode position
 * lookback:  how many blocks backward to prefetch (0 = just current)
 * lookahead: how many blocks forward to prefetch (0 = none)
 *
 * The prefetch is non-blocking — the CPU issues the hint and continues
 * execution. By the time attention needs the data, it's in L1/L2. */
void wubu_rope_prefetch_kv(wubu_kv_cacheline_t *store,
                            const int *block_ids, int n_blocks,
                            int pos, int lookback, int lookahead) {
    if (!store || !block_ids || n_blocks <= 0 || pos < 0) return;

    int tokens_per_block = store->block_size;
    int current_block = pos / tokens_per_block;

    /* Prefetch current block (likely already in cache but ensures it) */
    for (int h = 0; h < store->n_kv_heads; h++) {
        float *K = wubu_kv_cacheline_K(store, block_ids[current_block], h, 0);
        float *V = wubu_kv_cacheline_V(store, block_ids[current_block], h, 0);
        if (K) __builtin_prefetch(K, 0, 1);  /* read, low locality */
        if (V) __builtin_prefetch(V, 0, 1);
    }

    /* Prefetch lookback blocks */
    for (int d = 1; d <= lookback; d++) {
        int blk = current_block - d;
        if (blk < 0 || blk >= n_blocks) continue;
        if (block_ids[blk] < 0) continue;
        for (int h = 0; h < store->n_kv_heads; h++) {
            float *K = wubu_kv_cacheline_K(store, block_ids[blk], h, 0);
            if (K) __builtin_prefetch(K, 0, 0);  /* read, no temporal locality */
        }
    }

    /* Prefetch lookahead blocks */
    for (int d = 1; d <= lookahead; d++) {
        int blk = current_block + d;
        if (blk < 0 || blk >= n_blocks) continue;
        if (block_ids[blk] < 0) continue;
        for (int h = 0; h < store->n_kv_heads; h++) {
            float *K = wubu_kv_cacheline_K(store, block_ids[blk], h, 0);
            if (K) __builtin_prefetch(K, 0, 0);
        }
    }
}

/* Compute the RoPE rotation angle for a given position and dimension.
 * This is a reference implementation for verifying prefetch locality. */
float wubu_rope_theta(int dim, int pos, int head_dim) {
    if (head_dim <= 0) return 0.0f;
    int freq_idx = dim % (head_dim / 2);
    float theta = pos * powf(10000.0f, -2.0f * freq_idx / (float)head_dim);
    return theta;
}
