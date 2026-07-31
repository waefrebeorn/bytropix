/*
 * wubu_fast_attn.h — Zero-malloc, precomputed-RoPE fast GQA attention.
 *
 * The existing wubu_gqa_forward() does ~175 malloc/free calls per layer.
 * This module replaces the hot attention inner loop with:
 *   1. Zero per-token malloc — pre-allocated workspace, reused
 *   2. Precomputed RoPE sin/cos tables — no powf/cosf/sinf in loop
 *   3. Direct cache pointer access — no dispatch abstraction
 *   4. Tiled AVX2-FMA dot product + online softmax
 *   5. Bandwidth-optimal K/V sequential streaming
 *
 * For 512K context, this eliminates ~175K malloc calls per decode step
 * and turns the RoPE cost from O(T * n_rot * 3 transcendentals) to O(0)
 * (precomputed at init).
 */

#ifndef WUBU_FAST_ATTN_H
#define WUBU_FAST_ATTN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Model config */
    int   n_q_heads;
    int   n_kv_heads;
    int   head_dim;
    int   max_ctx;
    int   n_rot;
    float freq_base;
    float scale_factor;
    int   kv_group_size;

    /* Tiling constants */
#define WUBU_TILE_Q      16
#define WUBU_STREAM_BLOCK 32

    /* Precomputed RoPE tables: [max_ctx, n_rot/2] */
    float *rope_sin;
    float *rope_cos;

    /* Workspace (zero per-token malloc) */
    float *attn_scores;   /* [n_q_heads * max_ctx] — scores + softmax weights */
    float *k_buf;         /* [n_kv_heads * head_dim] — token K read buffer */
    float *v_buf;         /* [n_kv_heads * head_dim] — token V read buffer */
} wubu_fast_attn_ctx_t;

/* Init workspace. Returns NULL on OOM. Call once per model. */
wubu_fast_attn_ctx_t *wubu_fast_attn_init(
        int n_q_heads, int n_kv_heads, int head_dim,
        int max_ctx, int n_rot, float freq_base, float scale_factor);

/* Free workspace */
void wubu_fast_attn_free(wubu_fast_attn_ctx_t *ctx);

/* Apply RoPE to q and k using precomputed tables.
 * q: [n_q_heads * head_dim] (modified in-place)
 * k: [n_kv_heads * head_dim] (modified in-place)
 * pos: absolute position */
void wubu_fast_attn_rope(wubu_fast_attn_ctx_t *ctx,
                              float *q, float *k, int pos);

/* Fast single-token decode attention (N=1).
 * Computes out = softmax(Q·K^T / sqrt(d)) · V using pre-allocated workspace.
 * Q must already have RoPE applied. K/V are read directly from F32 cache.
 *
 * q:        [n_q_heads * head_dim] — already RoPE'd
 * k_cache:  [cache_len, n_kv_heads, head_dim] F32, contiguous
 * v_cache:  [cache_len, n_kv_heads, head_dim] F32, contiguous
 * cache_len: number of cached positions
 * out:      [n_q_heads * head_dim] — output
 * n_threads: OpenMP thread count (1 = serial)
 */
void wubu_fast_attn_decode(wubu_fast_attn_ctx_t *ctx,
                                const float *q,
                                const float *k_cache,
                                const float *v_cache,
                                int cache_len,
                                float *out,
                                int n_threads);

/* Write K/V to cache directly (memcpy, no dispatch).
 * k_new: [n_kv_heads * head_dim] — already RoPE'd
 * v_new: [n_kv_heads * head_dim]
 * k_cache: [max_ctx, n_kv_heads, head_dim] F32
 * v_cache: [max_ctx, n_kv_heads, head_dim] F32
 * pos: absolute position to write */
void wubu_fast_attn_write_kv(wubu_fast_attn_ctx_t *ctx,
                                  const float *k_new,
                                  const float *v_new,
                                  float *k_cache, float *v_cache,
                                  int pos);

/* Singleton accessor — lazily init one context per model config. */
wubu_fast_attn_ctx_t *wubu_fast_attn_get_ctx(
        int n_q_heads, int n_kv_heads, int head_dim,
        int n_rot, float freq_base, float scale_factor);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_FAST_ATTN_H */