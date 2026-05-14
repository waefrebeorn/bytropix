/**
 * hashmind_model.c — Full forward + backward pass for HashMind transformer
 *
 * Architecture: Embed → N×(Attn + FFN) → Output
 * Full manual backprop with QKV attention gradients.
 * 36,296 parameters total.
 */
#include "hashmind_model.h"
#include "rolling_hash.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

long hashmind_param_count(void) {
    long c = VOCAB * D_MODEL;                           /* token_embed */
    c += D_MODEL;                                        /* hash_projector */
    for (int l = 0; l < N_LAYERS; l++) {
        c += D_MODEL * D_MODEL * 3;                      /* qkv_w */
        c += D_MODEL * D_MODEL;                          /* out_w */
        c += D_MODEL * D_FF;                             /* ffn1_w */
        c += D_FF * D_MODEL;                             /* ffn2_w */
        c += D_MODEL * 2;                                /* ln1 */
        c += D_MODEL * 2;                                /* ln2 */
    }
    c += D_MODEL * VOCAB;                                /* out_w */
    return c;
}

void hashmind_model_init(HashMindModel* model) {
    srand(42);
    float scale = 0.02f;
    for (int i = 0; i < VOCAB; i++)
        for (int j = 0; j < D_MODEL; j++)
            model->token_embed[i][j] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * scale;
    for (int j = 0; j < D_MODEL; j++)
        model->hash_projector[j] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * scale;
    for (int l = 0; l < N_LAYERS; l++) {
        float s2 = sqrtf(2.0f / D_MODEL);
        for (int i = 0; i < D_MODEL; i++) {
            for (int j = 0; j < D_MODEL * 3; j++)
                model->blocks[l].qkv_w[i][j] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * s2;
            for (int j = 0; j < D_MODEL; j++)
                model->blocks[l].out_w[i][j] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * s2;
            for (int j = 0; j < D_FF; j++)
                model->blocks[l].ffn1_w[i][j] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * s2;
        }
        float sf = sqrtf(2.0f / D_FF);
        for (int i = 0; i < D_FF; i++)
            for (int j = 0; j < D_MODEL; j++)
                model->blocks[l].ffn2_w[i][j] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * sf;
        for (int j = 0; j < D_MODEL; j++) {
            model->blocks[l].ln1_gamma[j] = 1.0f;
            model->blocks[l].ln1_beta[j] = 0.0f;
            model->blocks[l].ln2_gamma[j] = 1.0f;
            model->blocks[l].ln2_beta[j] = 0.0f;
        }
    }
    float so = sqrtf(2.0f / D_MODEL);
    for (int i = 0; i < D_MODEL; i++)
        for (int j = 0; j < VOCAB; j++)
            model->out_w[i][j] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * so;
}

/* ─── Forward pass ─── */
void hashmind_forward(HashMindModel* model,
                       const uint32_t* context_hashes, int num_hashes,
                       const int* context_indices, int ctx_len,
                       float* logits_out, BlockActs* acts) {
    (void)num_hashes;
    float x[D_MODEL];
    int cur = ctx_len - 1;
    if (cur < 0) cur = 0;

    /* Dual-source embedding */
    for (int j = 0; j < D_MODEL; j++)
        x[j] = model->token_embed[context_indices[cur]][j];
    float hval = (float)context_hashes[cur] / (float)MODULUS;
    for (int j = 0; j < D_MODEL; j++)
        x[j] += hval * model->hash_projector[j];
    for (int j = 0; j < D_MODEL; j += 2) {
        float div = expf(-(float)j * logf(10000.0f) / D_MODEL);
        float p = (float)ctx_len;
        x[j] += sinf(p * div);
        if (j + 1 < D_MODEL) x[j+1] += cosf(p * div);
    }
    if (acts) for (int j = 0; j < D_MODEL; j++) acts->x[j] = x[j];

    for (int l = 0; l < N_LAYERS; l++) {
        /* Layer norm 1 */
        float ln1_out[D_MODEL];
        float m1 = 0, v1 = 0;
        for (int j = 0; j < D_MODEL; j++) m1 += x[j];
        m1 /= D_MODEL;
        for (int j = 0; j < D_MODEL; j++) v1 += (x[j] - m1) * (x[j] - m1);
        v1 /= D_MODEL;
        for (int j = 0; j < D_MODEL; j++)
            ln1_out[j] = nn_layer_norm(x[j], m1, v1, model->blocks[l].ln1_gamma[j], model->blocks[l].ln1_beta[j]);
        if (acts) for (int j = 0; j < D_MODEL; j++) acts->ln1_out[j] = ln1_out[j];

        /* QKV */
        float q[N_HEADS][D_HEAD], k[N_HEADS][D_HEAD], vv[N_HEADS][D_HEAD];
        for (int h = 0; h < N_HEADS; h++) {
            int hd = h * D_HEAD;
            for (int d = 0; d < D_HEAD; d++) {
                int qi = hd + d, ki = D_MODEL + hd + d, vi = 2*D_MODEL + hd + d;
                q[h][d] = 0; k[h][d] = 0; vv[h][d] = 0;
                for (int j = 0; j < D_MODEL; j++) {
                    q[h][d]  += ln1_out[j] * model->blocks[l].qkv_w[j][qi];
                    k[h][d]  += ln1_out[j] * model->blocks[l].qkv_w[j][ki];
                    vv[h][d] += ln1_out[j] * model->blocks[l].qkv_w[j][vi];
                }
            }
        }
        if (acts) {
            for (int h = 0; h < N_HEADS; h++)
                for (int d = 0; d < D_HEAD; d++) {
                    acts->q[h][d] = q[h][d];
                    acts->k[h][d] = k[h][d];
                    acts->v[h][d] = vv[h][d];
                }
        }

        /* Attention: single token */
        float ac[D_MODEL];
        for (int j = 0; j < D_MODEL; j++) ac[j] = 0;
        for (int h = 0; h < N_HEADS; h++) {
            float s = 0;
            for (int d = 0; d < D_HEAD; d++) s += q[h][d] * k[h][d];
            s /= sqrtf((float)D_HEAD);
            float aw = expf(s) / (expf(s) + 1e-7f);
            if (acts) { acts->attn_scores[h] = s; acts->attn_weights[h] = aw; }
            int hd = h * D_HEAD;
            for (int d = 0; d < D_HEAD; d++)
                ac[hd + d] = aw * vv[h][d];
        }
        if (acts) for (int j = 0; j < D_MODEL; j++) acts->attn_concat[j] = ac[j];

        /* Out proj */
        float attn_out[D_MODEL];
        for (int j = 0; j < D_MODEL; j++) {
            attn_out[j] = 0;
            for (int i = 0; i < D_MODEL; i++)
                attn_out[j] += ac[i] * model->blocks[l].out_w[i][j];
        }

        /* Residual 1 */
        float r1[D_MODEL];
        for (int j = 0; j < D_MODEL; j++) r1[j] = x[j] + attn_out[j];
        if (acts) for (int j = 0; j < D_MODEL; j++) acts->residual1[j] = r1[j];

        /* Layer norm 2 */
        float ln2_out[D_MODEL];
        float m2 = 0, v2 = 0;
        for (int j = 0; j < D_MODEL; j++) m2 += r1[j];
        m2 /= D_MODEL;
        for (int j = 0; j < D_MODEL; j++) v2 += (r1[j] - m2) * (r1[j] - m2);
        v2 /= D_MODEL;
        for (int j = 0; j < D_MODEL; j++)
            ln2_out[j] = nn_layer_norm(r1[j], m2, v2, model->blocks[l].ln2_gamma[j], model->blocks[l].ln2_beta[j]);
        if (acts) for (int j = 0; j < D_MODEL; j++) acts->ln2_out[j] = ln2_out[j];

        /* FFN */
        float ffn_h[D_FF], ffn_r[D_FF];
        for (int j = 0; j < D_FF; j++) {
            ffn_h[j] = 0;
            for (int i = 0; i < D_MODEL; i++)
                ffn_h[j] += ln2_out[i] * model->blocks[l].ffn1_w[i][j];
        }
        if (acts) for (int j = 0; j < D_FF; j++) acts->ffn_hidden[j] = ffn_h[j];
        for (int j = 0; j < D_FF; j++) {
            ffn_r[j] = ffn_h[j] > 0 ? ffn_h[j] : 0;
            if (acts) acts->ffn_relu[j] = ffn_r[j];
        }
        float ffn_out[D_MODEL];
        for (int j = 0; j < D_MODEL; j++) {
            ffn_out[j] = 0;
            for (int i = 0; i < D_FF; i++)
                ffn_out[j] += ffn_r[i] * model->blocks[l].ffn2_w[i][j];
        }

        /* Residual 2 */
        for (int j = 0; j < D_MODEL; j++) x[j] = r1[j] + ffn_out[j];
    }

    for (int j = 0; j < VOCAB; j++) {
        logits_out[j] = 0;
        for (int i = 0; i < D_MODEL; i++)
            logits_out[j] += x[i] * model->out_w[i][j];
        /* Clamp logits to prevent overflow in softmax */
        if (logits_out[j] > 20.0f) logits_out[j] = 20.0f;
        if (logits_out[j] < -20.0f) logits_out[j] = -20.0f;
    }
}

/* ─── Backward pass (full manual autograd) ─── */
void hashmind_backward(HashMindModel* model, HashMindGrad* grad,
                        const float* dlogits,
                        const uint32_t* context_hashes, int num_hashes,
                        const int* context_indices, int ctx_len,
                        const BlockActs* acts) {
    (void)num_hashes;
    int cur = ctx_len - 1;
    if (cur < 0) cur = 0;
    float hval = (float)context_hashes[cur] / (float)MODULUS;

    float dx[D_MODEL];
    /* dL/d(out_w) = acts->x ⊗ dlogits; dL/dx_final = dlogits · out_w^T */
    for (int i = 0; i < D_MODEL; i++) {
        dx[i] = 0;
        for (int j = 0; j < VOCAB; j++) {
            grad->out_w[i][j] += acts->x[i] * dlogits[j];
            dx[i] += model->out_w[i][j] * dlogits[j];
        }
    }

    for (int l = N_LAYERS - 1; l >= 0; l--) {
        /* dx is dL/d(residual2) = dL/dx_after_block */

        /* ── FFN sub-layer ── */
        /* dx_ffn_out = dx (from residual2: r1 + ffn_out = x, so dL/dffn_out = dx) */

        /* dL/dffn2_w[i][j] = ffn_relu[i] * dx[j] */
        for (int i = 0; i < D_FF; i++)
            for (int j = 0; j < D_MODEL; j++)
                grad->blocks[l].ffn2_w[i][j] += acts->ffn_relu[i] * dx[j];

        /* dL/dffn_relu[i] = sum_j(dx[j] * ffn2_w[i][j]) */
        float dffn[D_FF];
        for (int i = 0; i < D_FF; i++) {
            dffn[i] = 0;
            for (int j = 0; j < D_MODEL; j++)
                dffn[i] += dx[j] * model->blocks[l].ffn2_w[i][j];
        }

        /* Through ReLU: dL/dffn_hidden = dffn * (ffn_hidden > 0) */
        for (int i = 0; i < D_FF; i++) {
            if (acts->ffn_hidden[i] <= 0) dffn[i] = 0;
        }

        /* dL/dffn1_w[i][j] = ln2_out[i] * dffn[j] */
        for (int i = 0; i < D_MODEL; i++)
            for (int j = 0; j < D_FF; j++)
                grad->blocks[l].ffn1_w[i][j] += acts->ln2_out[i] * dffn[j];

        /* dL/dln2_out[i] = sum_j(dffn[j] * ffn1_w[i][j]) */
        float dln2[D_MODEL];
        for (int i = 0; i < D_MODEL; i++) {
            dln2[i] = 0;
            for (int j = 0; j < D_FF; j++)
                dln2[i] += dffn[j] * model->blocks[l].ffn1_w[i][j];
        }

        /* dL/dr1 += d_ln2_x (from norm path) + dx (from residual skip) */
        /* Layer norm 2 backward */
        float m2 = 0, v2 = 0;
        for (int j = 0; j < D_MODEL; j++) m2 += acts->residual1[j];
        m2 /= D_MODEL;
        for (int j = 0; j < D_MODEL; j++) v2 += (acts->residual1[j] - m2) * (acts->residual1[j] - m2);
        v2 /= D_MODEL;
        float inv_s2 = 1.0f / sqrtf(v2 + 1e-5f);
        float sdy = 0, sdn = 0;
        for (int j = 0; j < D_MODEL; j++) { sdy += dln2[j]; sdn += dln2[j] * (acts->residual1[j] - m2); }
        float in = 1.0f / D_MODEL;
        float dr1[D_MODEL];
        for (int j = 0; j < D_MODEL; j++) {
            grad->blocks[l].ln2_gamma[j] += dln2[j] * (acts->residual1[j] - m2) * inv_s2;
            grad->blocks[l].ln2_beta[j]  += dln2[j];
            float g = (D_MODEL * dln2[j] - sdy - (acts->residual1[j] - m2) * inv_s2 * inv_s2 * sdn) * in;
            dr1[j] = model->blocks[l].ln2_gamma[j] * g * inv_s2;
        }
        for (int j = 0; j < D_MODEL; j++)
            dr1[j] += dx[j];  /* residual skip */

        /* ── Attention sub-layer ── */
        /* dr1 = dL/d(r1) = dL/d(x + attn_out), we need dL/d(attn_out) */
        /* dL/dattn_out = dr1 (this is the derivative w.r.t. attn_out of r1 = x + attn_out) */
        /* But also dL/dx = dr1 (the skip connection path) — we'll set dx for next block */

        /* dL/d(attn_concat) = dr1 · out_w^T */
        float da[D_MODEL];
        for (int i = 0; i < D_MODEL; i++) {
            da[i] = 0;
            for (int j = 0; j < D_MODEL; j++)
                da[i] += dr1[j] * model->blocks[l].out_w[i][j];
        }

        /* dL/dout_w[i][j] = acts->attn_concat[i] * dr1[j] */
        for (int i = 0; i < D_MODEL; i++)
            for (int j = 0; j < D_MODEL; j++)
                grad->blocks[l].out_w[i][j] += acts->attn_concat[i] * dr1[j];

        /* Through attention: per-head */
        for (int h = 0; h < N_HEADS; h++) {
            int hd = h * D_HEAD;
            float aw = acts->attn_weights[h];
            /* dL/ds = sum_d( da[hd+d] * v[h][d] ) * aw * (1-aw) */
            float ds = 0;
            for (int d = 0; d < D_HEAD; d++)
                ds += da[hd + d] * acts->v[h][d];
            ds *= aw * (1.0f - aw);
            ds /= sqrtf((float)D_HEAD);

            for (int d = 0; d < D_HEAD; d++) {
                /* dL/dv[h][d] = da[hd+d] * aw */
                float dv = da[hd + d] * aw;
                /* dL/dq[h][d] += ds * k[h][d] */
                float dq_d = ds * acts->k[h][d];
                /* dL/dk[h][d] += ds * q[h][d] */
                float dk_d = ds * acts->q[h][d];

                int qi = hd + d, ki = D_MODEL + hd + d, vi = 2*D_MODEL + hd + d;
                for (int j = 0; j < D_MODEL; j++) {
                    grad->blocks[l].qkv_w[j][qi] += acts->ln1_out[j] * dq_d;
                    grad->blocks[l].qkv_w[j][ki] += acts->ln1_out[j] * dk_d;
                    grad->blocks[l].qkv_w[j][vi] += acts->ln1_out[j] * dv;
                }
            }
        }

        /* dL/dqkv_w propagation: we need dL/dln1_out */
        /* dL/dln1_out[j] = sum_{h,d}( qkv_w[j,qi] * dq_d + qkv_w[j,ki] * dk_d + qkv_w[j,vi] * dv ) */
        float dqkv_sum[D_MODEL];
        for (int j = 0; j < D_MODEL; j++) dqkv_sum[j] = 0;

        for (int h = 0; h < N_HEADS; h++) {
            int hd = h * D_HEAD;
            float aw = acts->attn_weights[h];
            float ds_score = 0;
            for (int d2 = 0; d2 < D_HEAD; d2++)
                ds_score += da[hd + d2] * acts->v[h][d2];
            ds_score *= aw * (1.0f - aw) / sqrtf((float)D_HEAD);

            for (int d = 0; d < D_HEAD; d++) {
                float dq = ds_score * acts->k[h][d];
                float dk = ds_score * acts->q[h][d];
                float dv = da[hd + d] * aw;

                int qi = hd + d, ki = D_MODEL + hd + d, vi = 2*D_MODEL + hd + d;
                for (int j = 0; j < D_MODEL; j++) {
                    dqkv_sum[j] += model->blocks[l].qkv_w[j][qi] * dq
                                 + model->blocks[l].qkv_w[j][ki] * dk
                                 + model->blocks[l].qkv_w[j][vi] * dv;
                }
            }
        }

        /* Layer norm 1 backward: dL/dln1_out is dqkv_sum + dr1 (from skip) */
        /* Actually ln1_out feeds into QKV, and the residual skip is x -> r1 */
        /* dL/dx (pre-norm) = dr1 (skip) + dln1_out from norm backward */
        float m1 = 0, v1 = 0;
        for (int j = 0; j < D_MODEL; j++) m1 += acts->x[j];
        m1 /= D_MODEL;
        for (int j = 0; j < D_MODEL; j++) v1 += (acts->x[j] - m1) * (acts->x[j] - m1);
        v1 /= D_MODEL;
        float inv_s1 = 1.0f / sqrtf(v1 + 1e-5f);
        float sdy1 = 0, sdn1 = 0;
        for (int j = 0; j < D_MODEL; j++) { sdy1 += dqkv_sum[j]; sdn1 += dqkv_sum[j] * (acts->x[j] - m1); }
        float in1 = 1.0f / D_MODEL;
        float dln1_gamma[D_MODEL], dln1_beta[D_MODEL];
        for (int j = 0; j < D_MODEL; j++) {
            dln1_gamma[j] = dqkv_sum[j] * (acts->x[j] - m1) * inv_s1;
            dln1_beta[j] = dqkv_sum[j];
            float g = (D_MODEL * dqkv_sum[j] - sdy1 - (acts->x[j] - m1) * inv_s1 * inv_s1 * sdn1) * in1;
            dx[j] = model->blocks[l].ln1_gamma[j] * g * inv_s1 + dr1[j];  /* skip + norm path */
            grad->blocks[l].ln1_gamma[j] += dln1_gamma[j];
            grad->blocks[l].ln1_beta[j] += dln1_beta[j];
        }
    }

    /* ─── Embedding gradients ─── */
    for (int j = 0; j < D_MODEL; j++) {
        grad->token_embed[context_indices[cur]][j] += dx[j];
        grad->hash_projector[j] += dx[j] * hval;
    }
}

/* ─── Apply gradients: WubuOptimizer (toroidal decomposition + Adam) ───
 *
 * From WuBu_TgT_Test.py: decomposes gradient into remainder (mod 2π, stable direction)
 * and quotient (magnitude wraps). momentum uses remainder, adaptive LR uses raw gradient.
 * This prevents NaN/blowup at singularities without information loss.
 * moment1 from wrapped gradient (stable direction), moment2 from raw gradient (adaptive LR).
 */
void hashmind_apply_gradients(TrainCtx* ctx) {
    float lr = ctx->lr;
    float wd = ctx->weight_decay;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    HashMindModel* m = ctx->model;
    HashMindGrad* g = ctx->grad;
    HashMindMomentum* v = ctx->vel;
    float* mp = (float*)m;
    float* gp = (float*)g;
    float* vp = (float*)v;
    long n = hashmind_param_count();
    float pi_f = 3.14159265f;
    float boundary = 2.0f * pi_f;

    ctx->step++;
    float t = (float)ctx->step;
    float bc1 = 1.0f - powf(beta1, t);  /* bias correction 1 */
    float bc2 = 1.0f - powf(beta2, t);  /* bias correction 2 */
    if (bc1 < 1e-8f) bc1 = 1e-8f;
    if (bc2 < 1e-8f) bc2 = 1e-8f;

    for (long i = 0; i < n; i++) {
        float raw_g = gp[i]; gp[i] = 0;
        if (raw_g == 0.0f) continue;

        /* Clamp raw gradient to prevent extreme values */
        if (raw_g > 100.0f) raw_g = 100.0f;
        if (raw_g < -100.0f) raw_g = -100.0f;

        /* Toroidal remainder: mod into [-pi, pi] for stable direction.
         * This is the part that gives stable momentum direction regardless of magnitude. */
        float wrapped = fmodf(raw_g + pi_f, boundary);
        if (wrapped < 0) wrapped += boundary;
        wrapped -= pi_f;

        /* moment1 from remainder (stable direction) — stored in first half of vp */
        vp[i] = beta1 * vp[i] + (1.0f - beta1) * wrapped;

        /* moment2 from raw gradient (adaptivity) — stored in second half of vp */
        /* We shift index by n to get a second buffer */
        vp[i + n] = beta2 * vp[i + n] + (1.0f - beta2) * (raw_g * raw_g);

        /* Bias-corrected moments */
        float m1_hat = vp[i] / bc1;
        float m2_hat = vp[i + n] / bc2;

        /* Adam update with weight decay (decoupled) */
        float update = -lr * m1_hat / (sqrtf(m2_hat) + eps) - lr * wd * mp[i];

        /* Safety clamp */
        const float max_upd = 0.1f;
        if (update > max_upd) update = max_upd;
        if (update < -max_upd) update = -max_upd;

        mp[i] += update;
    }
}

void hashmind_zero_grad(HashMindGrad* grad) {
    memset(grad, 0, sizeof(HashMindGrad));
}

/* ─── Generate next token ─── */
int hashmind_generate(HashMindModel* model,
                       const int* indices, int len,
                       const uint32_t* hashes, float temperature) {
    int ctx_len = len < CONTEXT_LEN ? len : CONTEXT_LEN;
    float logits[VOCAB];
    BlockActs acts;
    hashmind_forward(model, hashes + len - ctx_len, ctx_len,
                     indices + len - ctx_len, ctx_len, logits, &acts);

    if (temperature > 0) {
        float maxv = logits[0];
        for (int i = 1; i < VOCAB; i++) if (logits[i] > maxv) maxv = logits[i];
        float sum = 0, probs[VOCAB];
        for (int i = 0; i < VOCAB; i++) { probs[i] = expf((logits[i] - maxv) / temperature); sum += probs[i]; }
        float r = (float)rand() / RAND_MAX;
        float cum = 0;
        for (int i = 0; i < VOCAB; i++) { cum += probs[i] / sum; if (r <= cum) return i; }
        return VOCAB - 1;
    } else {
        int best = 0;
        for (int i = 1; i < VOCAB; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
}

/* ─── Save/Load ─── */
int hashmind_save(const HashMindModel* model, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    size_t n = fwrite(model, 1, sizeof(HashMindModel), f);
    fclose(f);
    return (n == sizeof(HashMindModel)) ? 0 : -1;
}

int hashmind_load(HashMindModel* model, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    size_t n = fread(model, 1, sizeof(HashMindModel), f);
    fclose(f);
    return (n == sizeof(HashMindModel)) ? 0 : -1;
}
