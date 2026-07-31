/*
 * wubu_kv_cacheline.h -- Cache-line-aligned KV page storage (doc C03/I03).
 *
 * Allocates K/V pages aligned to 64-byte cache lines, eliminating
 * split cache-line loads during decode attention.
 *
 * Self-contained C11, no third-party deps.
 */

#ifndef WUBU_KV_CACHELINE_H
#define WUBU_KV_CACHELINE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int    block_size;   /* tokens per block */
    int    head_dim;     /* dimension per head */
    int    n_kv_heads;   /* number of KV heads */
    int    n_blocks;     /* total physical blocks */
    size_t block_bytes;  /* bytes per block (K or V only) */
    size_t total_bytes;  /* total bytes allocated for K (or V) */
    float *K_data;       /* cache-line-aligned K storage */
    float *V_data;       /* cache-line-aligned V storage */
} wubu_kv_cacheline_t;

/* Create aligned KV store. Returns NULL on failure. */
wubu_kv_cacheline_t *wubu_kv_cacheline_create(int block_size, int head_dim,
                                                int n_kv_heads, int n_blocks);
void wubu_kv_cacheline_free(wubu_kv_cacheline_t *store);

/* Get pointer to K/V data for (block, head, token_in_block). */
float *wubu_kv_cacheline_K(wubu_kv_cacheline_t *store, int block_id,
                            int head, int token_in_block);
float *wubu_kv_cacheline_V(wubu_kv_cacheline_t *store, int block_id,
                            int head, int token_in_block);

/* Write/read K/V vectors for a token. */
void wubu_kv_cacheline_write(wubu_kv_cacheline_t *store, int block_id,
                              int token_in_block,
                              const float *k_vec, const float *v_vec);
void wubu_kv_cacheline_read(wubu_kv_cacheline_t *store, int block_id,
                             int token_in_block,
                             float *out_k, float *out_v);

/* Verify alignment. */
bool wubu_kv_cacheline_is_aligned(wubu_kv_cacheline_t *store, int block_id);

/* Storage stats. */
void wubu_kv_cacheline_stats(wubu_kv_cacheline_t *store,
                              size_t *total_bytes, size_t *block_bytes,
                              int *n_blocks);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_KV_CACHELINE_H */
