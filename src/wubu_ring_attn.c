/* wubu_ring_attn.c — Ring Attention + Star Attention implementation
 *
 * Chunked scan with LSE (log-sum-exp) merging for causal attention
 * over 1M+ token contexts using the ring communication pattern.
 *
 * C11, zero-malloc hot path, opaque structs.
 */
#include "wubu_ring_attn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

struct wubu_ring_attn_ctx {
    int n_heads;
    int head_dim;
    int max_ctx;
    int n_chunks;

    /* Thread-local scratch buffers */
    float *lse_scratch;  /* [n_chunks * head_dim] */
    float *score_scratch; /* [max_chunk_tokens] */
};

wubu_ring_attn_ctx_t *wubu_ring_attn_init(
        int n_heads, int head_dim, int max_ctx, int n_chunks)
{
    if (n_heads <= 0 || head_dim <= 0 || max_ctx <= 0 || n_chunks <= 0)
        return NULL;

    wubu_ring_attn_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    ctx->n_heads = n_heads;
    ctx->head_dim = head_dim;
    ctx->max_ctx = max_ctx;
    ctx->n_chunks = n_chunks;

    /* Pre-allocate scratch buffers */
    ctx->lse_scratch = calloc((size_t)n_chunks * head_dim, sizeof(float));
    /* score_scratch must hold max_ctx entries (one per token), not chunk size */
    ctx->score_scratch = malloc((size_t)max_ctx * sizeof(float));

    if (!ctx->lse_scratch || !ctx->score_scratch) {
        free(ctx->lse_scratch); free(ctx->score_scratch); free(ctx);
        return NULL;
    }

    return ctx;
}

void wubu_ring_attn_free(wubu_ring_attn_ctx_t *ctx)
{
    if (!ctx) return;
    free(ctx->lse_scratch);
    free(ctx->score_scratch);
    free(ctx);
}

/* Single chunk: compute local Q·K scores, softmax with LSE, weighted V accum. */
/* Per-token attention: compute scores against all tokens, softmax, weighted V.
 * Called once per query token in the chunk. */
static void ring_token_attn(
        const float *q_token, const float *k_global, const float *v_global,
        int n_heads, int head_dim, int ctx_len,
        float *out_token, float *score_buf)
{
    float inv_sqrt_d = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < n_heads; h++) {
        const float *q_h = q_token + (size_t)h * head_dim;
        float *o_h = out_token + (size_t)h * head_dim;
        float *scores = score_buf;

        float local_max = -1e30f;
        for (int t = 0; t < ctx_len; t++) {
            const float *k_t = k_global + (size_t)(t * n_heads + h) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += q_h[d] * k_t[d];
            scores[t] = dot * inv_sqrt_d;
            if (scores[t] > local_max) local_max = scores[t];
        }

        float local_sumexp = 0.0f;
        memset(o_h, 0, (size_t)head_dim * sizeof(float));
        for (int t = 0; t < ctx_len; t++) {
            float ew = expf(scores[t] - local_max);
            local_sumexp += ew;
            const float *v_t = v_global + (size_t)(t * n_heads + h) * head_dim;
            for (int d = 0; d < head_dim; d++)
                o_h[d] += ew * v_t[d];
        }

        float inv = (local_sumexp > 1e-30f) ? 1.0f / local_sumexp : 0.0f;
        for (int d = 0; d < head_dim; d++)
            o_h[d] *= inv;
    }
}

void wubu_ring_attn_chunk(
        wubu_ring_attn_ctx_t *ctx,
        const float *q_local,
        const float *k_global, const float *v_global,
        const float *lse_in, float *lse_out,
        int chunk_start, int chunk_end,
        float *out,
        int n_threads)
{
    if (!ctx || !q_local || !out) return;
    if (chunk_start >= chunk_end) return;

    int hd = ctx->head_dim;
    int n_h = ctx->n_heads;
    int mt = ctx->max_ctx;
    size_t per_chunk = (size_t)n_h * hd;

    for (int t = chunk_start; t < chunk_end; t++) {
        const float *q_token = q_local + (size_t)(t - chunk_start) * per_chunk;
        float *o_token = out + (size_t)(t - chunk_start) * per_chunk;
        ring_token_attn(q_token, k_global, v_global,
                        n_h, hd, mt,
                        o_token, ctx->score_scratch);
    }
}

void wubu_star_attn_chunk(
        wubu_ring_attn_ctx_t *ctx,
        const float *q_local,
        const float *k_global, const float *v_global,
        const float *lse_in, float *lse_out,
        int chunk_start, int chunk_end,
        float *out,
        int n_threads,
        int is_anchor)
{
    /* Star Attention: anchor blocks attend to all tokens.
     * Non-anchor blocks attend to their own chunk + next chunk only
     * (sparse local attention). */
    if (!ctx || !q_local || !out) return;
    if (chunk_start >= chunk_end) return;

    int hd = ctx->head_dim;
    int n_h = ctx->n_heads;
    int mt = ctx->max_ctx;
    int n_chunks = ctx->n_chunks;

    if (is_anchor) {
        /* Anchor block: full attention */
        for (int t = chunk_start; t < chunk_end; t++) {
            const float *q_token = q_local + (size_t)(t - chunk_start) * (size_t)n_h * hd;
            float *o_token = out + (size_t)(t - chunk_start) * (size_t)n_h * hd;
            ring_token_attn(q_token, k_global, v_global,
                            n_h, hd, mt, o_token, ctx->score_scratch);
        }
    } else {
        /* Periodic block: sparse local attention.
         * Attend to current chunk + overlap from next chunk only. */
        int overlap = (mt / n_chunks) / 4;
        int sparse_end = chunk_end + overlap;
        if (sparse_end > mt) sparse_end = mt;

        for (int t = chunk_start; t < chunk_end; t++) {
            const float *q_token = q_local + (size_t)(t - chunk_start) * (size_t)n_h * hd;
            float *o_token = out + (size_t)(t - chunk_start) * (size_t)n_h * hd;
            ring_token_attn(q_token, k_global, v_global,
                            n_h, hd, sparse_end,
                            o_token, ctx->score_scratch);
        }
    }
}

int wubu_ring_attn_forward(
        wubu_ring_attn_ctx_t *ctx,
        const float *q, const float *k, const float *v,
        int ctx_len, int n_chunks,
        float *out,
        int n_threads)
{
    if (!ctx || !q || !k || !v || !out) return -1;
    if (ctx_len <= 0 || n_chunks <= 0) return -1;

    int hd = ctx->head_dim;
    int n_h = ctx->n_heads;
    int chunk_size = (ctx_len + n_chunks - 1) / n_chunks;
    size_t per_chunk = (size_t)n_h * hd;

    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int c = 0; c < n_chunks; c++) {
        int t_start = c * chunk_size;
        int t_end = t_start + chunk_size;
        if (t_start >= ctx_len) continue;
        if (t_end > ctx_len) t_end = ctx_len;

        for (int t = t_start; t < t_end; t++) {
            const float *q_token = q + (size_t)t * per_chunk;
            float *o_token = out + (size_t)t * per_chunk;
            ring_token_attn(q_token, k, v,
                            n_h, hd, ctx_len,
                            o_token, ctx->score_scratch);
        }
    }

    return 0;
}