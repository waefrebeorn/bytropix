/*
 * wubu_mmrope.c — 3D Multimodal RoPE implementation for MiniMax H3.
 *
 * H3-Omni-Transformer uses 3D MM-RoPE to encode positional relationships
 * across temporal and two spatial dimensions (t, h, w).
 *
 * The head dimension is split into 3 equal segments:
 *   - First 1/3:  temporal RoPE (position varies by frame)
 *   - Middle 1/3: spatial height RoPE (position varies by row)
 *   - Last 1/3:   spatial width RoPE (position varies by column)
 *
 * Each segment uses standard rotary position embedding:
 *   For dimension pair (i, i + d/2):
 *     x_i = x_i * cos(pos/theta^(2i/d)) - x_{i+d/2} * sin(pos/theta^(2i/d))
 *     x_{i+d/2} = x_i * sin(pos/theta^(2i/d)) + x_{i+d/2} * cos(pos/theta^(2i/d))
 *
 * where theta = theta_t for temporal segment, theta_h for height, theta_w for width.
 *
 * C11, self-contained, opaque struct.
 */
#include "wubu_mmrope.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

struct wubu_mmrope_ctx {
    int head_dim;
    int seg_dim;            /* head_dim / 3 */
    float theta_t, theta_h, theta_w;
    int num_tok_t, num_tok_h, num_tok_w;
    const int *pos_t, *pos_h, *pos_w;
};

wubu_mmrope_t *wubu_mmrope_init(int head_dim,
                                float theta_t, float theta_h, float theta_w,
                                int num_tok_t, int num_tok_h, int num_tok_w,
                                const int *pos_t, const int *pos_h, const int *pos_w)
{
    if (head_dim <= 0 || head_dim % 3 != 0)
        return NULL;  /* head_dim must be divisible by 3 for 3D RoPE */
    if (head_dim % 2 != 0)
        return NULL;  /* each segment must be even for complex rotation */
    if (!pos_t || !pos_h || !pos_w)
        return NULL;

    wubu_mmrope_t *ctx = (wubu_mmrope_t *)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    ctx->head_dim   = head_dim;
    ctx->seg_dim    = head_dim / 3;
    ctx->theta_t    = theta_t;
    ctx->theta_h    = theta_h;
    ctx->theta_w    = theta_w;
    ctx->num_tok_t  = num_tok_t;
    ctx->num_tok_h  = num_tok_h;
    ctx->num_tok_w  = num_tok_w;
    ctx->pos_t = pos_t;
    ctx->pos_h = pos_h;
    ctx->pos_w = pos_w;

    return ctx;
}

void wubu_mmrope_apply(const wubu_mmrope_t *ctx,
                       float *qk, int seq_len, int n_heads)
{
    if (!ctx || !qk || seq_len <= 0 || n_heads <= 0) return;

    int hd = ctx->head_dim;
    int sd = ctx->seg_dim;       /* dim per segment (t, h, w) */
    int half_sd = sd / 2;         /* each segment split into even/odd pairs */

    for (int s = 0; s < seq_len; s++) {
        int pt = ctx->pos_t[s];
        int ph = ctx->pos_h[s];
        int pw = ctx->pos_w[s];

        for (int h = 0; h < n_heads; h++) {
            float *head = qk + (size_t)s * n_heads * hd + (size_t)h * hd;

            /* --- Segment 0: Temporal RoPE (dim 0 .. sd-1) --- */
            for (int i = 0; i < half_sd; i++) {
                float freq = (float)(pt) / powf(ctx->theta_t, (float)(2 * i) / (float)sd);
                float c = cosf(freq);
                float sn = sinf(freq);
                float x0 = head[i];
                float x1 = head[i + half_sd];
                head[i]           = x0 * c - x1 * sn;
                head[i + half_sd] = x0 * sn + x1 * c;
            }

            /* --- Segment 1: Spatial height RoPE (dim sd .. 2*sd-1) --- */
            for (int i = 0; i < half_sd; i++) {
                float freq = (float)(ph) / powf(ctx->theta_h, (float)(2 * i) / (float)sd);
                float c = cosf(freq);
                float sn = sinf(freq);
                float x0 = head[sd + i];
                float x1 = head[sd + i + half_sd];
                head[sd + i]           = x0 * c - x1 * sn;
                head[sd + i + half_sd] = x0 * sn + x1 * c;
            }

            /* --- Segment 2: Spatial width RoPE (dim 2*sd .. 3*sd-1) --- */
            for (int i = 0; i < half_sd; i++) {
                float freq = (float)(pw) / powf(ctx->theta_w, (float)(2 * i) / (float)sd);
                float c = cosf(freq);
                float sn = sinf(freq);
                float x0 = head[2 * sd + i];
                float x1 = head[2 * sd + i + half_sd];
                head[2 * sd + i]           = x0 * c - x1 * sn;
                head[2 * sd + i + half_sd] = x0 * sn + x1 * c;
            }
        }
    }
}

void wubu_mmrope_close(wubu_mmrope_t *ctx)
{
    if (!ctx) return;
    free(ctx);
}
