/*
 * wubu_h3_norm.c — MiniMax H3 hyperbolic neural network normalization.
 *
 * Self-contained C11 module.
 *
 * H3 activation formula:
 *   gate = sigmoid(W_gate * x + b_gate)       // standard gate (SiLU/Swish)
 *   up   = tanh(W_up  * x + b_up)              // hyperbolic branch (H3 innovation)
 *   out  = gate * up                           // elementwise multiply
 *
 * The gate uses SiLU (swish) like standard LLaMA MLP.
 * The up   uses tanh instead of SiLU — this is what makes H3 "hyperbolic".
 *
 * NF4 integration:
 *   If the gate/up weights are NF4-quantized (dtype == ST_DTYPE_NF4),
 *   we dequantize them from the safetensors shard on-demand using
 *   nf4_dequantize_row + companion scale tensor lookup.
 *
 * Memory management:
 *   - If weights are pre-dequantized (F32), ctx->weight points to the
 *     caller's buffer — no copy, no free.
 *   - If NF4, a workspace buffer is used for per-row dequantization.
 *
 * C11: no VLAs, no compound literals, opaque struct, minimal includes.
 * Angel Coder: self-contained, reusable, properly split from gguf_reader.
 */
#include "wubu_h3_norm.h"
#include "wubu_dequant_nf4.h"
#include "safetensors_reader.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

struct wubu_h3_norm_ctx {
    /* Weight data — either direct F32 pointers (caller-owned) or NULL
     * (NF4 path uses dequant workspace below). */
    const float  *gate_w;       /* [out_dim, in_dim] */
    const float  *gate_b;       /* [out_dim] or NULL */
    const float  *up_w;         /* [out_dim, in_dim] — tanh branch */
    const float  *up_b;         /* [out_dim] or NULL */

    /* NF4 raw sources (optional — used if F32 ptrs are NULL). */
    const uint8_t *gate_raw;    /* NF4 packed codes */
    const uint8_t *up_raw;
    float gate_scale;
    float up_scale;
    long  in_dim;
    long  out_dim;
    int   nf4;                  /* 1 = weights are NF4 packed */

    /* Working buffers (heap-allocated per layer). */
    float *buf_gate;            /* dequantized gate row (out_dim) */
    float *buf_up;              /* dequantized up row (out_dim) */
    float *buf_x;               /* projected input rows (out_dim) */
};

wubu_h3_norm_t *wubu_h3_norm_init(const float *gate_w, const float *gate_b,
                                  const float *up_w, const float *up_b,
                                  long in_dim, long out_dim)
{
    if (!gate_w || !up_w || in_dim <= 0 || out_dim <= 0) return NULL;

    wubu_h3_norm_t *ctx = (wubu_h3_norm_t *)calloc(1, sizeof(wubu_h3_norm_t));
    if (!ctx) return NULL;

    ctx->gate_w = gate_w;
    ctx->gate_b = gate_b;
    ctx->up_w   = up_w;
    ctx->up_b   = up_b;
    ctx->in_dim  = in_dim;
    ctx->out_dim = out_dim;
    ctx->nf4     = 0;

    /* Pre-allocate working buffers: each output row needs projection temps. */
    ctx->buf_gate = (float *)malloc((size_t)out_dim * sizeof(float));
    ctx->buf_up   = (float *)malloc((size_t)out_dim * sizeof(float));
    ctx->buf_x    = (float *)malloc((size_t)out_dim * sizeof(float));

    if (!ctx->buf_gate || !ctx->buf_up || !ctx->buf_x) {
        wubu_h3_norm_close(ctx);
        return NULL;
    }

    return ctx;
}

wubu_h3_norm_t *wubu_h3_norm_init_nf4(const uint8_t *gate_raw, float gate_scale,
                                     const uint8_t *up_raw, float up_scale,
                                     long in_dim, long out_dim)
{
    if (!gate_raw || !up_raw || in_dim <= 0 || out_dim <= 0) return NULL;

    wubu_h3_norm_t *ctx = (wubu_h3_norm_t *)calloc(1, sizeof(wubu_h3_norm_t));
    if (!ctx) return NULL;

    ctx->gate_raw   = gate_raw;
    ctx->up_raw     = up_raw;
    ctx->gate_scale = gate_scale;
    ctx->up_scale   = up_scale;
    ctx->in_dim     = in_dim;
    ctx->out_dim    = out_dim;
    ctx->nf4        = 1;

    /* NF4 workspace: dequantized row of size out_dim each. */
    ctx->buf_gate = (float *)malloc((size_t)out_dim * sizeof(float));
    ctx->buf_up   = (float *)malloc((size_t)out_dim * sizeof(float));
    ctx->buf_x    = (float *)malloc((size_t)out_dim * sizeof(float));

    if (!ctx->buf_gate || !ctx->buf_up || !ctx->buf_x) {
        wubu_h3_norm_close(ctx);
        return NULL;
    }

    return ctx;
}

void wubu_h3_norm_apply(const wubu_h3_norm_t *ctx,
                        const float *x,   /* [in_dim] */
                        float *out)        /* [out_dim] */
{
    if (!ctx || !x || !out) return;
    long in_dim  = ctx->in_dim;
    long out_dim = ctx->out_dim;

    if (ctx->nf4) {
        /* NF4 path: dequantize one row of gate/up weights at a time, then matmul.
         * NF4 packs 2 elements per byte; row bytes = ceil(in_dim / 2). */
        long row_bytes = (in_dim + 1) / 2;
        /* Workspace: one dequantized row of size in_dim. */
        float *wk = (float *)malloc((size_t)in_dim * sizeof(float));
        if (!wk) return;

        for (long o = 0; o < out_dim; o++) {
            /* Dequant gate row o */
            const uint8_t *grow = ctx->gate_raw + (size_t)o * row_bytes;
            nf4_dequantize_row(grow, wk, ctx->gate_scale, in_dim);
            /* Compute gate[o] = W_gate[o] · x + b_gate[o] */
            float g = 0.0f;
            for (long i = 0; i < in_dim; i++) g += wk[i] * x[i];
            if (ctx->gate_b) g += ctx->gate_b[o];
            ctx->buf_x[o] = g;

            /* Dequant up row o */
            const uint8_t *urow = ctx->up_raw + (size_t)o * row_bytes;
            nf4_dequantize_row(urow, wk, ctx->up_scale, in_dim);
            /* Compute up[o] = W_up[o] · x + b_up[o] */
            float u = 0.0f;
            for (long i = 0; i < in_dim; i++) u += wk[i] * x[i];
            if (ctx->up_b) u += ctx->up_b[o];
            ctx->buf_gate[o] = g;
            ctx->buf_up[o]   = u;
        }
        free(wk);
    } else {
        /* F32 path: direct matrix-vector multiply. */
        for (long o = 0; o < out_dim; o++) {
            float g = 0.0f, u = 0.0f;
            const float *gw = ctx->gate_w + (size_t)o * in_dim;
            const float *uw = ctx->up_w   + (size_t)o * in_dim;
            for (long i = 0; i < in_dim; i++) {
                float xi = x[i];
                g += gw[i] * xi;
                u += uw[i] * xi;
            }
            if (ctx->gate_b) g += ctx->gate_b[o];
            if (ctx->up_b)   u += ctx->up_b[o];
            ctx->buf_gate[o] = g;
            ctx->buf_up[o]   = u;
        }
    }

    /* H3 activation: gate = SiLU(g), up = tanh(u), out = gate * up */
    for (long o = 0; o < out_dim; o++) {
        float g = ctx->buf_gate[o];
        float u = ctx->buf_up[o];
        /* SiLU: x * sigmoid(x) */
        float sig = 1.0f / (1.0f + expf(-g));
        float silu = g * sig;
        /* Tanh (the H3 "hyperbolic" branch) */
        float tanh_u = tanf(u);
        out[o] = silu * tanh_u;
    }
}

void wubu_h3_norm_close(wubu_h3_norm_t *ctx)
{
    if (!ctx) return;
    free(ctx->buf_gate);
    free(ctx->buf_up);
    free(ctx->buf_x);
    free(ctx);
}
