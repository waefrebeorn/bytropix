/* wubu_ring_attn.h — Ring Attention for 1M+ token contexts (C11)
 *
 * Implements Chunked Scan attention (Liu et al. 2023, Ring Attention):
 *   - Distributes KV cache across multiple devices/chunks
 *   - Each device computes local attention + running LSE (log-sum-exp)
 *   - Ring communication: pass KV + LSE around the ring
 *   - Final output = softmax(LSE_global) * weighted_V_sum
 *
 * Also implements Star Attention pattern (Acharya et al. 2024):
 *   - Anchor blocks + periodic blocks for long-context efficiency
 *
 * Zero-malloc design. Opaque struct. Self-contained C11 module.
 */
#ifndef WUBU_RING_ATTN_H
#define WUBU_RING_ATTN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_ring_attn_ctx wubu_ring_attn_ctx_t;

/* Initialize ring attention context.
 * n_heads:      total attention heads
 * head_dim:     per-head dimension
 * max_ctx:      maximum context length (1M+)
 * n_chunks:     number of ring chunks (1 = no ring, n_chunks = n_devices) */
wubu_ring_attn_ctx_t *wubu_ring_attn_init(
        int n_heads, int head_dim, int max_ctx, int n_chunks);

void wubu_ring_attn_free(wubu_ring_attn_ctx_t *ctx);

/* Process one chunk of the ring.
 * q_local:     [local_tokens, head_dim] query for this chunk
 * k_global:    [max_ctx, head_dim] full KV cache (read-only, all chunks)
 * v_global:    [max_ctx, head_dim] full V cache (read-only, all chunks)
 * lse_in:      [n_chunks, head_dim] incoming LSE from previous ring step
 * lse_out:     [n_chunks, head_dim] outgoing LSE to next ring step
 * chunk_start: start token index of this chunk
 * chunk_end:   end token index of this chunk
 * out:         [local_tokens, head_dim] output for this chunk
 * n_threads:   OpenMP thread count */
void wubu_ring_attn_chunk(
        wubu_ring_attn_ctx_t *ctx,
        const float *q_local,
        const float *k_global, const float *v_global,
        const float *lse_in, float *lse_out,
        int chunk_start, int chunk_end,
        float *out,
        int n_threads);

/* Star Attention style: anchor blocks get full attention,
 * periodic blocks get sparse attention. */
void wubu_star_attn_chunk(
        wubu_ring_attn_ctx_t *ctx,
        const float *q_local,
        const float *k_global, const float *v_global,
        const float *lse_in, float *lse_out,
        int chunk_start, int chunk_end,
        float *out,
        int n_threads,
        int is_anchor); /* 1 = anchor block, full attention; 0 = periodic, sparse */

/* One-shot ring attention (single forward, multi-chunk).
 * Returns 0 on success, -1 on error. */
int wubu_ring_attn_forward(
        wubu_ring_attn_ctx_t *ctx,
        const float *q, const float *k, const float *v,
        int ctx_len, int n_chunks,
        float *out,
        int n_threads);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_RING_ATTN_H */
