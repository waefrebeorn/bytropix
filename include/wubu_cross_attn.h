/*
 * wubu_cross_attn.h — Cross-attention for multimodal fusion
 *
 * Enables attention between different modalities (text↔vision, text↔audio).
 * Q comes from one modality (e.g., text decoder), K/V from another
 * (e.g., vision encoder output). Uses the same fast decode path as
 * wubu_fast_attn but with separate K/V caches from encoder outputs.
 *
 * Zero-malloc, precomputed, C11 only. No god headers.
 */
#ifndef WUBU_CROSS_ATTN_H
#define WUBU_CROSS_ATTN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque context — defined in implementation */
typedef struct wubu_cross_attn_ctx wubu_cross_attn_ctx_t;

/* Initialize cross-attention context.
 * n_q_heads:    query attention heads (from decoder)
 * n_kv_heads:   key/value attention heads (from encoder)
 * head_dim:     per-head dimension (must be same for Q and K/V)
 * max_kv_len:   max encoder sequence length (for K/V cache prealloc)
 * Returns NULL on OOM. */
wubu_cross_attn_ctx_t *wubu_cross_attn_init(
        int n_q_heads, int n_kv_heads, int head_dim, int max_kv_len);

void wubu_cross_attn_free(wubu_cross_attn_ctx_t *ctx);

/* Store encoder K/V (called once after encoder forward pass).
 * k_enc: [enc_len, n_kv_heads, head_dim] F32
 * v_enc: [enc_len, n_kv_heads, head_dim] F32
 * enc_len: actual encoder sequence length (≤ max_kv_len) */
void wubu_cross_attn_store_kv(
        wubu_cross_attn_ctx_t *ctx,
        const float *k_enc, const float *v_enc, int enc_len);

/* Cross-attention decode: Q (decoder) attends to stored encoder K/V.
 * q:     [n_q_heads * head_dim] — decoder query (already RoPE'd if needed)
 * out:   [n_q_heads * head_dim] — cross-attention output
 * n_threads: OpenMP threads
 *
 * Uses split-K parallelism when enc_len is large. */
void wubu_cross_attn_decode(
        wubu_cross_attn_ctx_t *ctx,
        const float *q,
        float *out,
        int n_threads);

/* Q8 compressed cross-attention: store encoder K/V in Q8 format.
 * Reduces encoder KV cache memory by 4× vs F32. */
void wubu_cross_attn_store_kv_q8(
        wubu_cross_attn_ctx_t *ctx,
        const float *k_enc, const float *v_enc, int enc_len);

void wubu_cross_attn_decode_q8(
        wubu_cross_attn_ctx_t *ctx,
        const float *q,
        float *out,
        int n_threads);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_CROSS_ATTN_H */
