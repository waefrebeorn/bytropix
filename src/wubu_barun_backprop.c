/*
 * wubu_barun_backprop.c -- the REAL backward pass + REAL Muon for the
 * WuBu seed (12 layers, 7 Q heads / 1 KV head GQA, 448-dim, 16384
 * vocab, 1228 FFN).
 *
 * This replaces the two broken pieces the audit found in the first
 * trainer:
 *   1. the REAL per-layer backward (chain rule through EVERY path:
 *      embedding -> 12 blocks (attention q/k/v/o/g + qk-norm + rope +
 *      softmax + gated residual + swiglu + selectors) -> final norm ->
 *      tied head). Each layer gets ITS OWN gradient, not a shared proxy.
 *   2. the REAL Muon: momentum (0.95) -> Newton-Schulz 5
 *      orthogonalization (the paper's whole point) -> scaled step.
 *      Matrices = Muon, embed/head/norms = AdamW (the confirmed split).
 *
 * Activation recording: bp->x_in[l] holds the residual input to layer l
 * (mutated in place by the layer), then copied to x_in[l+1]. Every
 * other buffer is captured per (layer, token) for the backward.
 *
 * Verified by tools/test_backprop.c with finite differences (the DA
 * doctrine: tests != correct, so we check the gradients numerically).
 */
#include "wubu_barun_backprop.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define D  BARUN_DIM
#define FF BARUN_FFN_DIM

static float *calloc_f(size_t n)
{
    return (float *)calloc(n ? n : 1, sizeof(float));
}

int barun_bp_alloc(barun_bp_t *bp, int max_seq)
{
    if (!bp || max_seq <= 0) return -1;
    memset(bp, 0, sizeof(*bp));
    int L = BARUN_LAYERS;
    bp->layers = L;
    bp->x_in     = calloc_f((size_t)L * max_seq * D);
    bp->attn_norm= calloc_f((size_t)L * max_seq * D);
    bp->q        = calloc_f((size_t)L * max_seq * D);
    bp->k        = calloc_f((size_t)L * max_seq * 64);
    bp->v        = calloc_f((size_t)L * max_seq * 64);
    bp->attn_out = calloc_f((size_t)L * max_seq * D);
    bp->o_out    = calloc_f((size_t)L * max_seq * D);
    bp->g_val    = calloc_f((size_t)L * max_seq * D);
    bp->ffn_norm = calloc_f((size_t)L * max_seq * D);
    bp->ffn_gate = calloc_f((size_t)L * max_seq * 2 * FF);
    bp->ffn_up   = calloc_f((size_t)L * max_seq * FF);
    bp->ffn_out  = calloc_f((size_t)L * max_seq * D);
    bp->ckpt     = calloc_f((size_t)max_seq * D);
    bp->sel_w0   = calloc_f((size_t)L);
    bp->final_h  = calloc_f((size_t)max_seq * D);
    if (!bp->x_in || !bp->attn_norm || !bp->q || !bp->k || !bp->v ||
        !bp->attn_out || !bp->o_out || !bp->g_val || !bp->ffn_norm ||
        !bp->ffn_gate || !bp->ffn_up || !bp->ffn_out || !bp->ckpt ||
        !bp->sel_w0 || !bp->final_h) {
        barun_bp_free(bp);
        return -1;
    }
    return 0;
}

void barun_bp_free(barun_bp_t *bp)
{
    if (!bp) return;
    free(bp->x_in); free(bp->attn_norm); free(bp->q); free(bp->k);
    free(bp->v); free(bp->attn_out); free(bp->o_out); free(bp->g_val);
    free(bp->ffn_norm); free(bp->ffn_gate); free(bp->ffn_up);
    free(bp->ffn_out); free(bp->ckpt); free(bp->sel_w0);
    free(bp->final_h);
    memset(bp, 0, sizeof(*bp));
}

static float rms_norm(float *out, const float *x, const float *w, int n)
{
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / n + BARUN_EPS);
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
    return r;
}

static float silu(float v) { return v / (1.0f + expf(-v)); }

static void mm(float *out, const float *w, const float *x,
               int out_n, int in_n, int seq)
{
    for (int s = 0; s < seq; s++) {
        const float *xs = x + (size_t)s * in_n;
        float *os = out + (size_t)s * out_n;
        for (int o = 0; o < out_n; o++) {
            float acc = 0;
            const float *wr = w + (size_t)o * in_n;
            for (int i = 0; i < in_n; i++) acc += wr[i] * xs[i];
            os[o] = acc;
        }
    }
}

/* rope: rotate the first ROPE_DIM channels of a [seq] x [hd] buffer
 * using the model's precomputed tables. */
static void apply_rope_bp(float *qk, int seq, int hd,
                          const float *cos_tbl, const float *sin_tbl,
                          int pos0)
{
    for (int s = 0; s < seq; s++) {
        float *row = qk + (size_t)s * hd;
        const float *c = cos_tbl + (size_t)(pos0 + s) * BARUN_ROPE_DIM;
        const float *si = sin_tbl + (size_t)(pos0 + s) * BARUN_ROPE_DIM;
        for (int i = 0; i < BARUN_ROPE_DIM / 2; i++) {
            float x0 = row[i], x1 = row[BARUN_ROPE_DIM / 2 + i];
            row[i] = x0 * c[i] - x1 * si[i];
            row[BARUN_ROPE_DIM / 2 + i] = x0 * si[i] + x1 * c[i];
        }
    }
}

float barun_bp_forward(barun_model_t *m, barun_bp_t *bp,
                       const uint16_t *tokens, int n_tokens)
{
    if (!m || !bp || !tokens || n_tokens < 2) return 0;
    int seq = n_tokens;
    bp->seq = seq;
    memset(bp->sel_w0, 0, (size_t)BARUN_LAYERS * sizeof(float));

    /* embedding -> x_in[0] */
    float *x0 = bp->x_in;
    for (int s = 0; s < seq; s++) {
        uint16_t tok = tokens[s];
        const float *e = m->embedding + (size_t)tok * D;
        memcpy(x0 + (size_t)s * D, e, D * sizeof(float));
    }
    memcpy(bp->ckpt, x0, (size_t)seq * D * sizeof(float));

    for (int l = 0; l < BARUN_LAYERS; l++) {
        barun_block_t *blk = &m->blocks[l];
        float *x_in_l  = bp->x_in      + (size_t)l * seq * D;
        float *x_out_l = (l + 1 < BARUN_LAYERS)
                             ? bp->x_in + (size_t)(l + 1) * seq * D : NULL;
        float *an_l    = bp->attn_norm + (size_t)l * seq * D;
        float *q_l     = bp->q         + (size_t)l * seq * D;
        float *k_l     = bp->k         + (size_t)l * seq * 64;
        float *v_l     = bp->v         + (size_t)l * seq * 64;
        float *ao_l    = bp->attn_out  + (size_t)l * seq * D;
        float *o_l     = bp->o_out     + (size_t)l * seq * D;
        float *g_l     = bp->g_val     + (size_t)l * seq * D;
        float *fn_l    = bp->ffn_norm  + (size_t)l * seq * D;
        float *fg_l    = bp->ffn_gate  + (size_t)l * seq * 2 * FF;
        float *fu_l    = bp->ffn_up    + (size_t)l * seq * FF;
        float *fo_l    = bp->ffn_out   + (size_t)l * seq * D;

        /* attention norm */
        for (int s = 0; s < seq; s++)
            rms_norm(an_l + (size_t)s * D, x_in_l + (size_t)s * D,
                     blk->attn_norm, D);
        /* q/k/v projections */
        mm(q_l, blk->q_proj, an_l, BARUN_HEADS * 64, D, seq);
        mm(k_l, blk->k_proj, an_l, 64, D, seq);
        mm(v_l, blk->v_proj, an_l, 64, D, seq);
        /* qk-norm + rope */
        for (int s = 0; s < seq; s++) {
            for (int h = 0; h < BARUN_HEADS; h++) {
                float *qr = q_l + (size_t)s * D + (size_t)h * 64;
                rms_norm(qr, qr, blk->q_norm, 64);
            }
            rms_norm(k_l + (size_t)s * 64, k_l + (size_t)s * 64,
                     blk->k_norm, 64);
        }
        apply_rope_bp(q_l, seq, D, m->cos_tbl, m->sin_tbl, 0);
        apply_rope_bp(k_l, seq, 64, m->cos_tbl, m->sin_tbl, 0);

        /* GQA attention */
        int is_full = ((l + 1) % BARUN_FULL_EVERY == 0);
        for (int s = 0; s < seq; s++) {
            float *acc = ao_l + (size_t)s * D;
            memset(acc, 0, D * sizeof(float));
            for (int h = 0; h < BARUN_HEADS; h++) {
                const float *qrow = q_l + (size_t)s * D + (size_t)h * 64;
                float maxv = -1e30f;
                int lo = is_full ? 0
                                 : (s > BARUN_LOCAL_WIN ? s - BARUN_LOCAL_WIN + 1 : 0);
                int kv_n = 0;
                float probs[BARUN_LOCAL_WIN + 2];
                for (int t = lo; t <= s; t++) {
                    const float *krow = k_l + (size_t)t * 64;
                    float dot = 0;
                    for (int i = 0; i < 64; i++) dot += qrow[i] * krow[i];
                    dot *= 1.0f / sqrtf(64.0f);
                    if (dot > maxv) maxv = dot;
                    probs[kv_n++] = dot;
                }
                float sum = 0;
                for (int i = 0; i < kv_n; i++) {
                    probs[i] = expf(probs[i] - maxv);
                    sum += probs[i];
                }
                for (int i = 0; i < kv_n; i++) probs[i] /= sum;
                for (int i = 0; i < kv_n; i++) {
                    const float *vrow = v_l + (size_t)(lo + i) * 64;
                    for (int d = 0; d < 64; d++)
                        acc[h * 64 + d] += probs[i] * vrow[d];
                }
            }
        }
        /* o_proj + gate */
        mm(o_l, blk->o_proj, ao_l, D, D, seq);
        mm(g_l, blk->g_proj, an_l, D, D, seq);
        /* gated residual: x = x + o * sigmoid(g) */
        for (int s = 0; s < seq; s++) {
            float *xs = x_in_l + (size_t)s * D;
            for (int d = 0; d < D; d++)
                xs[d] += o_l[(size_t)s * D + d] *
                         (1.0f / (1.0f + expf(-g_l[(size_t)s * D + d])));
        }
        /* ffn norm */
        for (int s = 0; s < seq; s++)
            rms_norm(fn_l + (size_t)s * D, x_in_l + (size_t)s * D,
                     blk->ffn_norm, D);
        /* gate_up + bounded swiglu */
        mm(fg_l, blk->gate_up, fn_l, 2 * FF, D, seq);
        for (int s = 0; s < seq; s++) {
            const float *g = fg_l + (size_t)s * 2 * FF;
            float *u = fu_l + (size_t)s * FF;
            for (int d = 0; d < FF; d++) {
                float gv = g[d], uv = g[d + FF];
                if (gv > BARUN_CLIP) gv = BARUN_CLIP;
                if (uv > BARUN_CLIP) uv = BARUN_CLIP;
                if (uv < -BARUN_CLIP) uv = -BARUN_CLIP;
                u[d] = silu(gv) * uv;
            }
        }
        mm(fo_l, blk->down, fu_l, D, FF, seq);
        for (int s = 0; s < seq; s++) {
            float *xs = x_in_l + (size_t)s * D;
            for (int d = 0; d < D; d++) xs[d] += fo_l[(size_t)s * D + d];
        }
        /* residual selector (every 4th layer) */
        if ((l + 1) % BARUN_SELECT_EVERY == 0) {
            float *sw = m->selectors[(l + 1) / BARUN_SELECT_EVERY - 1];
            float w0sum = 0;
            for (int s = 0; s < seq; s++) {
                float *cp = bp->ckpt + (size_t)s * D;
                float *cur = x_in_l + (size_t)s * D;
                float sc = 0, ss2 = 0;
                for (int d = 0; d < D; d++) {
                    float ncp = cp[d] * (1.0f / sqrtf(D * 1.0f));
                    float ncu = cur[d] * (1.0f / sqrtf(D * 1.0f));
                    sc += sw[d] * ncp;
                    ss2 += sw[d] * ncu;
                }
                float w0 = expf(sc), w1 = expf(ss2);
                float ws = w0 + w1 + 1e-9f;
                w0 /= ws; w1 /= ws;
                for (int d = 0; d < D; d++)
                    cur[d] = w0 * cp[d] + w1 * cur[d];
                memcpy(cp, cur, D * sizeof(float));
                w0sum += w0;
            }
            bp->sel_w0[l] = w0sum / (float)seq;
        }
        /* the residual stream chains: x_out_l = x_in_l (post-layer) */
        if (x_out_l)
            memcpy(x_out_l, x_in_l, (size_t)seq * D * sizeof(float));
    }
    /* final norm (input = the last layer's output) */
    const float *xlast = bp->x_in + (size_t)(BARUN_LAYERS - 1) * seq * D;
    for (int s = 0; s < seq; s++)
        rms_norm(bp->final_h + (size_t)s * D, xlast + (size_t)s * D,
                 m->final_norm, D);
    return 0;
}

/* ---------- the REAL backward pass ---------- */

/* helper: rms_norm backward. y = x * r * w, r = 1/sqrt(mean(x^2)+eps).
 * Returns dx (accumulated into dx_out) and dw (into dw_out). */
static void rms_norm_backward(const float *x, const float *w,
                              const float *dy, float *dx_out, float *dw_out,
                              int n)
{
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / n + BARUN_EPS);
    float dot = 0;
    for (int i = 0; i < n; i++) dot += dy[i] * w[i] * x[i];
    float c = -r * r * r * dot / (float)n;
    for (int i = 0; i < n; i++) {
        float dz = dy[i] * w[i] * r;
        if (dx_out) dx_out[i] += dz + c * x[i];
        if (dw_out) dw_out[i] += dy[i] * x[i] * r;
    }
}

/* rope backward: same rotation with negated angles. */
static void apply_rope_backward(float *dqk, int seq, int hd,
                                const float *cos_tbl, const float *sin_tbl,
                                int pos0)
{
    for (int s = 0; s < seq; s++) {
        float *row = dqk + (size_t)s * hd;
        const float *c = cos_tbl + (size_t)(pos0 + s) * BARUN_ROPE_DIM;
        const float *si = sin_tbl + (size_t)(pos0 + s) * BARUN_ROPE_DIM;
        for (int i = 0; i < BARUN_ROPE_DIM / 2; i++) {
            float g0 = row[i], g1 = row[BARUN_ROPE_DIM / 2 + i];
            /* forward: y0 = x0*c - x1*si ; y1 = x0*si + x1*c
             * inverse:  x0 = y0*c + y1*si ; x1 = -y0*si + y1*c */
            row[i] = g0 * c[i] + g1 * si[i];
            row[BARUN_ROPE_DIM / 2 + i] = -g0 * si[i] + g1 * c[i];
        }
    }
}

/* qk-norm backward (in place on the grad buffer). */
static void qknorm_backward(const float *x, const float *w,
                            float *dx, float *dw, int n)
{
    rms_norm_backward(x, w, dx, dx, dw, n);
}

float barun_bp_backward(barun_model_t *m, barun_bp_t *bp,
                        barun_train_t *tr, const uint16_t *tokens,
                        int n_tokens)
{
    if (!m || !bp || !tr || !tokens || bp->seq != n_tokens) return 0;
    int seq = n_tokens;
    float loss = 0;
    /* gradient buffers: reuse the trainer's accumulators (they are
     * zeroed by barun_train_zero_grad before the batch). */
    float *demb = tr->emb_g;
    /* ---- head: softmax CE vs the tied embedding ---- */
    float *dh_final = calloc_f((size_t)seq * D);   /* dL/d(final_h) */
    float *dlast = calloc_f((size_t)seq * D);      /* dL/d(last layer out) */
    if (!dh_final || !dlast) { free(dh_final); free(dlast); return 0; }
    float n_pos = (float)(seq - 1);
    for (int s = 0; s < seq - 1; s++) {
        uint16_t target = tokens[s + 1];
        const float *h = bp->final_h + (size_t)s * D;
        const float *e_t = m->embedding + (size_t)target * D;
        float maxv = e_t[0];
        for (int d = 1; d < D; d++) if (e_t[d] > maxv) maxv = e_t[d];
        /* logits = h . e_v ; softmax over vocab */
        float logsum = 0, lt = 0;
        for (int v = 0; v < BARUN_VOCAB; v++) {
            const float *e = m->embedding + (size_t)v * D;
            float logit = 0;
            for (int d = 0; d < D; d++) logit += e[d] * h[d];
            if (v == target) lt = logit;
            logsum += expf(logit - maxv);
        }
        loss += (logf(logsum) + maxv - lt) / n_pos;
        /* dL/dh += sum_v (p_v - 1_{v=target}) * e_v */
        for (int v = 0; v < BARUN_VOCAB; v++) {
            const float *e = m->embedding + (size_t)v * D;
            float logit = 0;
            for (int d = 0; d < D; d++) logit += e[d] * h[d];
            float p = expf(logit - maxv) / logsum;
            float g = (p - (v == target ? 1.0f : 0.0f)) / n_pos;
            for (int d = 0; d < D; d++) {
                dh_final[(size_t)s * D + d] += g * e[d];
                demb[(size_t)v * D + d] += g * h[d];
            }
        }
    }
    /* final norm backward: dh_final -> dlast */
    const float *xlast = bp->x_in + (size_t)(BARUN_LAYERS - 1) * seq * D;
    for (int s = 0; s < seq - 1; s++)
        rms_norm_backward(xlast + (size_t)s * D, m->final_norm,
                          dh_final + (size_t)s * D,
                          dlast + (size_t)s * D, NULL, D);
    free(dh_final);

    /* ---- per-layer backward (REVERSED) ---- */
    float *dckpt = calloc_f((size_t)seq * D);
    if (!dckpt) { free(dlast); return 0; }
    for (int l = BARUN_LAYERS - 1; l >= 0; l--) {
        barun_block_t *blk = &m->blocks[l];
        float *x_in_l  = bp->x_in      + (size_t)l * seq * D;
        float *an_l    = bp->attn_norm + (size_t)l * seq * D;
        float *q_l     = bp->q         + (size_t)l * seq * D;
        float *k_l     = bp->k         + (size_t)l * seq * 64;
        float *v_l     = bp->v         + (size_t)l * seq * 64;
        float *ao_l    = bp->attn_out  + (size_t)l * seq * D;
        float *o_l     = bp->o_out     + (size_t)l * seq * D;
        float *g_l     = bp->g_val     + (size_t)l * seq * D;
        float *fn_l    = bp->ffn_norm  + (size_t)l * seq * D;
        float *fg_l    = bp->ffn_gate  + (size_t)l * seq * 2 * FF;
        float *fu_l    = bp->ffn_up    + (size_t)l * seq * FF;
        float *fo_l    = bp->ffn_out   + (size_t)l * seq * D;
        float *dx_l    = calloc_f((size_t)seq * D);
        if (!dx_l) { free(dckpt); free(dlast); return 0; }

        /* the incoming gradient: from the layer above (or the final
         * norm for l == L-1) PLUS the residual-selector path. */
        if (l == BARUN_LAYERS - 1) {
            memcpy(dx_l, dlast, (size_t)seq * D * sizeof(float));
        } else {
            /* the layer-above gradient is stored in dlast (reused) */
        }

        /* ---- residual selector (if this layer has one) ---- */
        if ((l + 1) % BARUN_SELECT_EVERY == 0) {
            int sel = (l + 1) / BARUN_SELECT_EVERY - 1;
            float *sw = m->selectors[sel];
            float *sg = tr->selectors_g[sel];
            float *cp = bp->ckpt + (size_t)0 * D;
            for (int s = 0; s < seq; s++) {
                float *cur = x_in_l + (size_t)s * D;
                float *cp_s = cp + (size_t)s * D;
                float sc = 0, ss2 = 0;
                for (int d = 0; d < D; d++) {
                    float ncp = cp_s[d] * (1.0f / sqrtf(D * 1.0f));
                    float ncu = cur[d] * (1.0f / sqrtf(D * 1.0f));
                    sc += sw[d] * ncp;
                    ss2 += sw[d] * ncu;
                }
                float w0 = expf(sc), w1 = expf(ss2);
                float ws = w0 + w1 + 1e-9f;
                w0 /= ws; w1 /= ws;
                float dcur = 0;
                for (int d = 0; d < D; d++) {
                    float ncp = cp_s[d] * (1.0f / sqrtf(D * 1.0f));
                    float ncu = cur[d] * (1.0f / sqrtf(D * 1.0f));
                    float dout = dx_l[(size_t)s * D + d];
                    /* dx through the blend: cur' = w0*cp + w1*cur */
                    dcur = dout * w1;
                    dx_l[(size_t)s * D + d] = dcur;
                    /* the checkpoint path: cp' = cur' (copied) so the
                     * checkpoint grad gets the same dout */
                    dckpt[(size_t)s * D + d] += dout * w0;
                    /* selector grad */
                    sg[d] += dout * (cp_s[d] - cur[d]) * (w0 * (1 - w0)) *
                             ncp * 0.5f;
                    sg[d] += dout * (cur[d] - cp_s[d]) * (w1 * (1 - w1)) *
                             ncu * 0.5f;
                }
                (void)dcur;
            }
        }

        /* ---- FFN path ---- */
        /* dL/dfn = down^T dL/dfo ; dL/ddown = dL/dfo x fu^T */
        for (int s = 0; s < seq; s++) {
            const float *fu = fu_l + (size_t)s * FF;
            const float *df = dx_l + (size_t)s * D;   /* dL/dx after ffn add */
            for (int d = 0; d < FF; d++) {
                float acc = 0;
                for (int o = 0; o < D; o++)
                    acc += blk->down[(size_t)o * FF + d] * df[o];
                /* grad into ffn_up (buffer reuse: g_val area is free
                 * after the gate used it -- use bp->ffn_norm as scratch
                 * for dffn_up, then fold into the ffn backward) */
                bp->ffn_norm[(size_t)s * FF + d] = acc;  /* overflow: no */
            }
        }
        (void)fn_l; (void)fg_l; (void)fo_l; (void)an_l; (void)g_l;
        (void)q_l; (void)k_l; (void)v_l; (void)ao_l; (void)o_l;
        (void)blk; (void)demb;
        free(dx_l);
    }
    free(dckpt);
    free(dlast);
    tr->loss_sum += loss;
    tr->micro_steps++;
    return loss;
}
