/* lfm2_forward.c -- LFM2.5 forward orchestrator (C11, self-contained).
 * SPDX-License-Identifier: WaefreBeorn-UMV3 */
#include "lfm2_forward.h"
#include "lfm2_math.h"
#include "lfm2_conv.h"
#include "lfm2_attn.h"
#include "lfm2_ffn.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

bool lfm2_forward(const lfm2_model_t *m, const float *emb, int B, int T, float *logits) {
    if (B != 1) { fprintf(stderr, "lfm2: only B=1 supported\n"); return false; }
    int d = m->d_model;
    float *h = (float *)malloc((size_t)T * d * sizeof(float));
    memcpy(h, emb, (size_t)T * d * sizeof(float));
    /* HF Lfm2Model: embedding_norm is applied ONCE, AFTER all layers (not
     * before layer 0, not twice). Start the residual stream from raw embed. */
    float *scratch = (float *)malloc((size_t)T * d * sizeof(float));
    float *tmp = (float *)malloc((size_t)T * d * sizeof(float)); /* normalized input */

    for (int l = 0; l < m->n_layers; l++) {
        const lfm2_layer_t *L = &m->layers[l];

        /* operator path: tmp = operator_norm(h); op -> scratch; residual add */
        for (int t = 0; t < T; t++) {
            memcpy(tmp + (size_t)t * d, h + (size_t)t * d, d * sizeof(float));
            lfm2_rmsnorm(tmp + (size_t)t * d, L->op_norm, d, 1e-5f);
        }
        if (m->is_conv[l]) {
            lfm2_conv(L->in_proj, L->conv_w, L->out_proj, L->conv_k,
                      m->conv_dim, d, tmp, T, scratch);
        } else {
            float *kvc = m->kv_cache + (size_t)l * 2 * m->n_kv_heads * m->head_dim * m->kv_max_t;
            lfm2_gqa(L->q_proj, L->k_proj, L->v_proj, L->o_proj, L->q_ln, L->k_ln,
                     m->n_q_heads, m->n_kv_heads, m->head_dim, d, m->rope_theta,
                     tmp, T, kvc, m->kv_max_t, 0 /* start_pos: fresh prefill */, scratch);
        }
        for (size_t i = 0; i < (size_t)T * d; i++) h[i] += scratch[i];

        /* ffn path: tmp = ffn_norm(h); ffn -> scratch; residual add */
        for (int t = 0; t < T; t++) {
            memcpy(tmp + (size_t)t * d, h + (size_t)t * d, d * sizeof(float));
            lfm2_rmsnorm(tmp + (size_t)t * d, L->ffn_norm, d, 1e-5f);
        }
        lfm2_ffn(L->w1, L->w2, L->w3, m->ff_dim, d, tmp, T, scratch);
        for (size_t i = 0; i < (size_t)T * d; i++) h[i] += scratch[i];

        if (getenv("LFM2_DEBUG")) {
            const float *hp = h + (size_t)(T - 1) * d;
            float ss = 0.0f; for (int q = 0; q < d; q++) ss += hp[q] * hp[q];
            fprintf(stderr, "L%d h_norm=%.4f\n", l, sqrtf(ss / d));
        }
    }

    /* embedding_norm applied ONCE, after all layers (HF Lfm2Model) + tied lm_head */
    lfm2_rmsnorm(h + (size_t)(T - 1) * d, m->embed_norm, d, 1e-5f);
    lfm2_matmul_f32(h + (size_t)(T - 1) * d, m->embed, 1, d, m->vocab_size, logits);
    free(h); free(scratch); free(tmp);
    return true;
}
