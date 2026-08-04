/*
 * wubu_dsa.h -- DeepSeek-Sparse-Attention-style coarse-to-fine block
 * indexer (DSA indexer). Standalone C11; libc + libm only.
 */
#ifndef WUBU_DSA_H
#define WUBU_DSA_H

/* Opaque DSA indexer handle. */
typedef struct wubu_dsa wubu_dsa_t;

/*
 * Create a DSA indexer over a KV sequence split into n_blocks blocks of
 * block_size tokens each, head dimension d. top_k blocks are selected per
 * query (clamped to n_blocks at use time). Returns NULL on bad args
 * (any arg <= 0) or allocation failure.
 */
wubu_dsa_t *wubu_dsa_create(int n_blocks, int block_size, int top_k, int d);

/* Release the indexer. NULL-safe. */
void wubu_dsa_free(wubu_dsa_t *dsa);

/*
 * Coarse stage: score every block as dot(query, block_means[b]) and fill
 * out_blocks[0..k-1] with the top-k block indices sorted by score descending
 * (ties broken by lower block index), where k = min(top_k, n_blocks).
 * Returns k, or -1 on bad input (null pointers).
 */
int wubu_dsa_index(const wubu_dsa_t *dsa, const float *query,
                   const float *const *block_means, int *out_blocks);

/*
 * Coarse-to-fine: select the top-k blocks (block means computed from
 * block_keys), then softmax attention over ONLY the selected blocks' keys,
 * weighted into out[0..d_out-1]. out = softmax(dot/sqrt(d)) * V. Returns 0
 * on success, -1 on bad input.
 */
int wubu_dsa_attend(const wubu_dsa_t *dsa, const float *query,
                    const float *const *block_keys,
                    const float *const *block_vals, float *out, int d_out);

#endif /* WUBU_DSA_H */
