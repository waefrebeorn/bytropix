/*
 * wubu_cross_attn.c — Cross-attention for multimodal fusion
 *
 * Implements cross-attention decode where Q comes from the decoder
 * (text) and K/V come from an encoder (vision/audio). Uses the same
 * online softmax + split-K parallelism pattern as wubu_fast_attn.
 *
 * C11 only. No malloc on hot path. Opaque structs.
 */
#include "wubu_cross_attn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Q8 block layout (same as wubu_fast_attn.c) */
typedef struct { float d; int8_t qs[32]; } __attribute__((packed)) q8b_t;

struct wubu_cross_attn_ctx {
    int n_q_heads;
    int n_kv_heads;
    int head_dim;
    int max_kv_len;
    int kv_group_size; /* n_q / n_kv */
    int enc_len;       /* current stored encoder length */
    int is_q8;          /* 1 if KV stored in Q8 format */

    /* F32 K/V caches: [max_kv_len, n_kv_heads, head_dim] */
    float *k_cache;
    float *v_cache;

    /* Q8 K/V caches (when is_q8=1) */
    q8b_t *k_cache_q8;
    q8b_t *v_cache_q8;
    int kv_head_bytes_q8;
    int blocks_per_head;
};

wubu_cross_attn_ctx_t *wubu_cross_attn_init(
        int n_q_heads, int n_kv_heads, int head_dim, int max_kv_len)
{
    if (n_q_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_kv_len <= 0)
        return NULL;
    if (n_q_heads % n_kv_heads != 0) return NULL;

    wubu_cross_attn_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    ctx->n_q_heads = n_q_heads;
    ctx->n_kv_heads = n_kv_heads;
    ctx->head_dim = head_dim;
    ctx->max_kv_len = max_kv_len;
    ctx->kv_group_size = n_q_heads / n_kv_heads;
    ctx->enc_len = 0;
    ctx->is_q8 = 0;

    /* Pre-allocate F32 caches */
    size_t kv_size = (size_t)max_kv_len * n_kv_heads * head_dim * sizeof(float);
    ctx->k_cache = malloc(kv_size);
    ctx->v_cache = malloc(kv_size);
    if (!ctx->k_cache || !ctx->v_cache) {
        free(ctx->k_cache); free(ctx->v_cache); free(ctx);
        return NULL;
    }

    ctx->blocks_per_head = (head_dim + 31) / 32;
    ctx->kv_head_bytes_q8 = ctx->blocks_per_head * (int)sizeof(q8b_t);

    return ctx;
}

void wubu_cross_attn_free(wubu_cross_attn_ctx_t *ctx)
{
    if (!ctx) return;
    free(ctx->k_cache);
    free(ctx->v_cache);
    free(ctx->k_cache_q8);
    free(ctx->v_cache_q8);
    free(ctx);
}

void wubu_cross_attn_store_kv(
        wubu_cross_attn_ctx_t *ctx,
        const float *k_enc, const float *v_enc, int enc_len)
{
    if (!ctx || !k_enc || !v_enc || enc_len <= 0) return;
    if (enc_len > ctx->max_kv_len) enc_len = ctx->max_kv_len;
    ctx->enc_len = enc_len;
    ctx->is_q8 = 0;

    size_t bytes = (size_t)enc_len * ctx->n_kv_heads * ctx->head_dim * sizeof(float);
    memcpy(ctx->k_cache, k_enc, bytes);
    memcpy(ctx->v_cache, v_enc, bytes);
}

void wubu_cross_attn_store_kv_q8(
        wubu_cross_attn_ctx_t *ctx,
        const float *k_enc, const float *v_enc, int enc_len)
{
    if (!ctx || !k_enc || !v_enc || enc_len <= 0) return;
    if (enc_len > ctx->max_kv_len) enc_len = ctx->max_kv_len;
    ctx->enc_len = enc_len;
    ctx->is_q8 = 1;

    /* Allocate Q8 caches if not already */
    if (!ctx->k_cache_q8) {
        size_t sz = (size_t)ctx->max_kv_len * ctx->n_kv_heads * ctx->kv_head_bytes_q8;
        ctx->k_cache_q8 = malloc(sz);
        ctx->v_cache_q8 = malloc(sz);
        if (!ctx->k_cache_q8 || !ctx->v_cache_q8) return;
    }

    /* Quantize encoder K/V into Q8 blocks */
    int hd = ctx->head_dim;
    for (int t = 0; t < enc_len; t++) {
        for (int h = 0; h < ctx->n_kv_heads; h++) {
            const float *k_src = k_enc + (size_t)(t * ctx->n_kv_heads + h) * hd;
            const float *v_src = v_enc + (size_t)(t * ctx->n_kv_heads + h) * hd;
            q8b_t *k_dst = (q8b_t *)((char *)ctx->k_cache_q8 +
                (size_t)(t * ctx->n_kv_heads + h) * ctx->kv_head_bytes_q8);
            q8b_t *v_dst = (q8b_t *)((char *)ctx->v_cache_q8 +
                (size_t)(t * ctx->n_kv_heads + h) * ctx->kv_head_bytes_q8);

            for (int b = 0; b < ctx->blocks_per_head; b++) {
                float kmax = 0, vmax = 0;
                for (int i = 0; i < 32 && b*32+i < hd; i++) {
                    float ka = fabsf(k_src[b*32+i]), va = fabsf(v_src[b*32+i]);
                    if (ka > kmax) kmax = ka;
                    if (va > vmax) vmax = va;
                }
                if (kmax < 1e-10f) kmax = 1e-10f;
                if (vmax < 1e-10f) vmax = 1e-10f;
                k_dst[b].d = kmax / 127.0f;
                v_dst[b].d = vmax / 127.0f;
                for (int i = 0; i < 32; i++) {
                    int idx = b*32+i;
                    if (idx < hd) {
                        k_dst[b].qs[i] = (int8_t)(k_src[idx]/kmax*127.0f);
                        v_dst[b].qs[i] = (int8_t)(v_src[idx]/vmax*127.0f);
                    } else {
                        k_dst[b].qs[i] = 0;
                        v_dst[b].qs[i] = 0;
                    }
                }
            }
        }
    }
}

/* F32 cross-attention decode (split-K pattern) */
void wubu_cross_attn_decode(
        wubu_cross_attn_ctx_t *ctx,
        const float *q,
        float *out,
        int n_threads)
{
    if (!ctx || !q || !out || ctx->enc_len <= 0) {
        if (out) memset(out, 0, (size_t)(ctx ? ctx->n_q_heads * ctx->head_dim : 0) * sizeof(float));
        return;
    }

    int n_q = ctx->n_q_heads, n_kv = ctx->n_kv_heads;
    int hd = ctx->head_dim, gs = ctx->kv_group_size;
    int enc_len = ctx->enc_len;
    float inv_sqrt_hd = 1.0f / sqrtf((float)hd);

    /* Split-K: auto-detect splits from thread count */
    int n_splits = n_threads > 0 ? n_threads : 1;
    if (n_splits > enc_len) n_splits = enc_len;
    if (n_splits < 1) n_splits = 1;

    int ps = hd + 2;
    float *partials = (float *)alloca((size_t)n_splits * n_q * ps * sizeof(float));
    memset(partials, 0, (size_t)n_splits * n_q * ps * sizeof(float));

    int tps = (enc_len + n_splits - 1) / n_splits;

    #pragma omp parallel for num_threads(n_threads) collapse(2) schedule(dynamic)
    for (int split = 0; split < n_splits; split++) {
        for (int qh = 0; qh < n_q; qh++) {
            int g = qh / gs;
            int t0 = split * tps;
            int t1 = t0 + tps;
            if (t0 >= enc_len) continue;
            if (t1 > enc_len) t1 = enc_len;

            const float *q_h = q + (size_t)qh * hd;
            float *p = partials + (size_t)(split * n_q + qh) * ps;

            float lmax = -INFINITY;
            float lsum = 0.0f;

            for (int t = t0; t < t1; t++) {
                const float *k_h = ctx->k_cache + (size_t)(t * n_kv + g) * hd;
                const float *v_h = ctx->v_cache + (size_t)(t * n_kv + g) * hd;

                float dot = 0.0f;
                for (int d = 0; d < hd; d++) dot += q_h[d] * k_h[d];
                float s = dot * inv_sqrt_hd;

                if (s > lmax) {
                    float f = expf(lmax - s);
                    lsum = lsum * f + 1.0f;
                    for (int d = 0; d < hd; d++) p[2 + d] *= f;
                    lmax = s;
                }
                float ew = expf(s - lmax);
                lsum += ew;
                for (int d = 0; d < hd; d++) p[2 + d] += ew * v_h[d];
            }
            p[0] = lmax;
            p[1] = lsum;
        }
    }

    /* Merge */
    for (int qh = 0; qh < n_q; qh++) {
        float *oh = out + (size_t)qh * hd;
        float gmax = -INFINITY;
        for (int s = 0; s < n_splits; s++) {
            float *p = partials + (size_t)(s * n_q + qh) * ps;
            if (p[1] > 0.0f && p[0] > gmax) gmax = p[0];
        }
        if (gmax == -INFINITY) { memset(oh, 0, (size_t)hd * sizeof(float)); continue; }

        float gsum = 0.0f;
        for (int s = 0; s < n_splits; s++) {
            float *p = partials + (size_t)(s * n_q + qh) * ps;
            gsum += p[1] * expf(p[0] - gmax);
        }

        memset(oh, 0, (size_t)hd * sizeof(float));
        for (int s = 0; s < n_splits; s++) {
            float *p = partials + (size_t)(s * n_q + qh) * ps;
            float w = expf(p[0] - gmax) / (gsum + 1e-10f);
            for (int d = 0; d < hd; d++) oh[d] += w * p[2 + d];
        }
    }
}

/* Q8 cross-attention decode */
void wubu_cross_attn_decode_q8(
        wubu_cross_attn_ctx_t *ctx,
        const float *q,
        float *out,
        int n_threads)
{
    if (!ctx || !q || !out || ctx->enc_len <= 0 || !ctx->is_q8) {
        if (out) memset(out, 0, (size_t)(ctx ? ctx->n_q_heads * ctx->head_dim : 0) * sizeof(float));
        return;
    }

    int n_q = ctx->n_q_heads, n_kv = ctx->n_kv_heads;
    int hd = ctx->head_dim, gs = ctx->kv_group_size;
    int enc_len = ctx->enc_len, bph = ctx->blocks_per_head;
    int hbytes = ctx->kv_head_bytes_q8;
    float inv_sqrt_hd = 1.0f / sqrtf((float)hd);

    float *out_acc = (float *)alloca((size_t)n_q * hd * sizeof(float));
    memset(out_acc, 0, (size_t)n_q * hd * sizeof(float));
    float *dq = (float *)alloca((size_t)hd * sizeof(float));

    /* Per-head local max and sum_exp (same pattern as split-K decode) */
    float *h_max = (float *)alloca((size_t)n_q * sizeof(float));
    float *h_sum = (float *)alloca((size_t)n_q * sizeof(float));
    for (int qh = 0; qh < n_q; qh++) { h_max[qh] = -INFINITY; h_sum[qh] = 0.0f; }

    for (int t = 0; t < enc_len; t++) {
        for (int qh = 0; qh < n_q; qh++) {
            int g = qh / gs;
            const q8b_t *k_head = (const q8b_t *)
                ((const char *)ctx->k_cache_q8 + (size_t)t * n_kv * hbytes + (size_t)g * hbytes);
            const q8b_t *v_head = (const q8b_t *)
                ((const char *)ctx->v_cache_q8 + (size_t)t * n_kv * hbytes + (size_t)g * hbytes);

            const float *q_h = q + (size_t)qh * hd;
            float dot = 0.0f;
            for (int b = 0; b < bph; b++) {
                for (int i = 0; i < 32; i++) dq[i] = k_head[b].d * (float)k_head[b].qs[i];
                for (int i = 0; i < 32 && b*32+i < hd; i++)
                    dot += q_h[b*32+i] * dq[i];
            }
            float s = dot * inv_sqrt_hd;

            if (s > h_max[qh]) {
                    float f = expf(h_max[qh] - s);
                    h_sum[qh] = h_sum[qh] * f + 1.0f;
                    for (int d = qh*hd; d < (qh+1)*hd; d++) out_acc[d] *= f;
                    h_max[qh] = s;
                }
                float ew = expf(s - h_max[qh]);
                h_sum[qh] += ew;
                for (int b = 0; b < bph; b++) {
                    float d = v_head[b].d;
                    for (int i = 0; i < 32 && b*32+i < hd; i++) {
                        out_acc[qh*hd + b*32+i] += ew * d * (float)v_head[b].qs[i];
                    }
                }
        }
    }

    for (int qh = 0; qh < n_q; qh++) {
        float inv = 1.0f / (h_sum[qh] + 1e-10f);
        for (int d = 0; d < hd; d++)
            out[qh*hd + d] = out_acc[qh*hd + d] * inv;
    }
}
