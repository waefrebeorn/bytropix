/*
 * wubu_rope_prefetch.h -- RoPE-aware KV cache prefetch (doc A10).
 *
 * Software prefetch of KV blocks for nearby decode positions,
 * overlapped with current position's compute.
 *
 * Self-contained C11, no third-party deps.
 */

#ifndef WUBU_ROPE_PREFETCH_H
#define WUBU_ROPE_PREFETCH_H

#include "wubu_kv_cacheline.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Prefetch KV blocks for nearby positions (software prefetch).
 * Non-blocking — issues hint, continues execution. */
void wubu_rope_prefetch_kv(wubu_kv_cacheline_t *store,
                            const int *block_ids, int n_blocks,
                            int pos, int lookback, int lookahead);

/* Compute RoPE rotation angle for (dim, pos, head_dim). */
float wubu_rope_theta(int dim, int pos, int head_dim);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_ROPE_PREFETCH_H */
