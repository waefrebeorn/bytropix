/* wubu_lfm.c — LFM2.5-2.6B hybrid attention layer (C11, opaque, minimal)
 *
 * Hybrid architecture: linear attention (Gated DeltaNet / GLA) + softmax GQA
 * in alternating layers, combined via a learned interpolation gate.
 * Reuses wubu_linear_attn.h for the linear path; softmax GQA inline.
 *
 * The DeltaNet state is a d_model × d_model matrix (row-major), where
 * S[i][j] accumulates the outer-product (k_i * v_j) over past tokens.
 * wubu_deltanet_update takes S[d*d], k[d], v[d], and produces Sout[d*d].
 * The recurrent attention output for token t is: out = S_t * q (state @ query).
 */
#define _POSIX_C_SOURCE 200809L
#include "wubu_lfm.h"
#include "wubu_linear_attn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct wubu_lfm {
    wubu_lfm_cfg_t cfg;
};

wubu_lfm_t *wubu_lfm_create(const wubu_lfm_cfg_t *cfg) {
    if (!cfg || cfg->d_model <= 0 || cfg->n_heads <= 0 || cfg->d_head <= 0 ||
        cfg->n_kv_heads <= 0 || cfg->n_layers <= 0)
        return NULL;
    if (cfg->n_heads % cfg->n_kv_heads != 0) return NULL;
    wubu_lfm_t *lfm = (wubu_lfm_t *)calloc(1, sizeof(*lfm));
    if (!lfm) return NULL;
    lfm->cfg = *cfg;
    return lfm;
}

void wubu_lfm_free(wubu_lfm_t *lfm) {
    if (!lfm) return;
    free(lfm);
}

/* Linear attention path: Gated DeltaNet state update + output projection.
 * S: [d*d] state matrix (d_model × d_model, row-major)
 * k_lin: [d] key (projected input for this step)
 * v_lin: [d] value (projected input for this step)
 * beta: decay factor in [0,1]
 * Sout: [d*d] new state matrix
 * out:  [d]  attention output = Sout @ k_lin (state times key)
 *
 * The DeltaNet recurrence: S' = S - beta * (S·k - v) k^T
 * The attention output is the state applied to the query: out = S' · q
 * Here we use k_lin as both the key for the update and the query for output. */
int wubu_lfm_linear_attn(const float *S, const float *k_lin, const float *v_lin,
                          int d, float beta, float *Sout, float *out) {
    if (!S || !k_lin || !v_lin || !Sout || !out || d <= 0) return -1;
    /* Update state: Sout = S - beta*(S k - v) k^T */
    if (wubu_deltanet_update(S, k_lin, v_lin, d, beta, Sout) != 1)
        return -1;
    /* Output: out = Sout · k_lin (state matrix @ key vector) */
    for (int i = 0; i < d; i++) {
        float dot = 0;
        const float *row = Sout + (size_t)i * d;
        for (int j = 0; j < d; j++)
            dot += row[j] * k_lin[j];
        out[i] = dot;
    }
    return 0;
}

/* Standard softmax GQA attention (odd layers).
 * Causal masking: position pos can only attend to positions [0..pos].
 * KV heads are shared across head groups: n_heads / n_kv_heads heads
 * per KV head. */
int wubu_lfm_softmax_attn(const float *queries,
                           const float *keys, const float *values,
                           int n_heads, int n_kv_heads, int d_head,
                           int seq_len, int pos, float *out) {
    if (!queries || !keys || !values || !out) return -1;
    if (n_heads <= 0 || n_kv_heads <= 0 || d_head <= 0 || seq_len <= 0) return -1;
    if (pos < 0 || pos >= seq_len) return -1;
    if (n_heads % n_kv_heads != 0) return -1;

    int heads_per_kv = n_heads / n_kv_heads;

    for (int h = 0; h < n_heads; h++) {
        int kv_head = h / heads_per_kv;
        const float *q = queries + (size_t)h * d_head;
        const float *kv = keys + (size_t)kv_head * d_head * seq_len;
        const float *vv = values + (size_t)kv_head * d_head * seq_len;
        float *o = out + (size_t)h * d_head;

        float *scores = (float *)malloc((size_t)(pos + 1) * sizeof(float));
        if (!scores) return -1;

        float max_score = -1e30f;
        for (int i = 0; i <= pos; i++) {
            float dot = 0;
            const float *k_i = kv + (size_t)i * d_head;
            for (int j = 0; j < d_head; j++)
                dot += q[j] * k_i[j];
            dot /= sqrtf((float)d_head);
            scores[i] = dot;
            if (dot > max_score) max_score = dot;
        }

        float sum = 0;
        for (int i = 0; i <= pos; i++) {
            scores[i] = expf(scores[i] - max_score);
            sum += scores[i];
        }
        if (sum < 1e-8f) sum = 1e-8f;
        for (int i = 0; i <= pos; i++)
            scores[i] /= sum;

        for (int j = 0; j < d_head; j++)
            o[j] = 0;
        for (int i = 0; i <= pos; i++) {
            const float *v_i = vv + (size_t)i * d_head;
            for (int j = 0; j < d_head; j++)
                o[j] += scores[i] * v_i[j];
        }

        free(scores);
    }
    return 0;
}

/* Hybrid layer: combines linear + softmax attention.
 * output = gate * linear_out + (1 - gate) * softmax_out
 *
 * Even layers (layer_idx even): linear-dominant (gate > 0.5)
 * Odd layers: softmax-dominant (gate < 0.5)
 *
 * linear_state: [d_model * d_model] DeltaNet state matrix
 * query/key/value: [n_heads * d_head] projected inputs for softmax path
 * k_lin, v_lin: [d_model] for linear path
 * gate: interpolation weight [0,1]
 * layer_idx: 0-based layer index
 * out: [d_model]
 * Sout: [d_model * d_model] new linear state (caller manages)
 *
 * Returns 0 on success, -1 on error. */
int wubu_lfm_hybrid_layer(const wubu_lfm_t *lfm,
                           const float *linear_state,
                           const float *query, const float *key, const float *value,
                           const float *k_lin, const float *v_lin,
                           float gate, int layer_idx,
                           float *out, float *Sout) {
    if (!lfm || !linear_state || !query || !key || !value ||
        !k_lin || !v_lin || !out || !Sout) return -1;

    int d = lfm->cfg.d_model;
    int d_head = lfm->cfg.d_head;
    int n_heads = lfm->cfg.n_heads;
    int n_kv = lfm->cfg.n_kv_heads;

    /* Linear attention path: DeltaNet state update + output */
    float *linear_out = (float *)malloc((size_t)d * sizeof(float));
    if (!linear_out) return -1;
    /* beta = 0.9 (LFM2.5 typical decay) */
    if (wubu_lfm_linear_attn(linear_state, k_lin, v_lin, d, 0.9f,
                              Sout, linear_out) != 0) {
        free(linear_out);
        return -1;
    }

    /* Softmax attention path */
    float *softmax_out = (float *)calloc(n_heads * d_head, sizeof(float));
    if (!softmax_out) { free(linear_out); return -1; }
    /* Single token decode: seq_len=1, pos=0 */
    if (wubu_lfm_softmax_attn(query, key, value, n_heads, n_kv, d_head,
                              1, 0, softmax_out) != 0) {
        free(linear_out); free(softmax_out); return -1;
    }

    /* Combine via gate. Even layers weight linear more; odd layers softmax. */
    float g_linear, g_softmax;
    if (layer_idx % 2 == 0) {
        g_linear = gate;
        g_softmax = 1.0f - gate;
    } else {
        g_linear = 1.0f - gate;
        g_softmax = gate;
    }

    /* Both outputs are [d_model] — linear_out is [d], softmax_out is [n_heads*d_head].
     * If d == n_heads * d_head, they're the same size. Otherwise we truncate. */
    int out_dim = d;
    int sm_dim = n_heads * d_head;
    int min_dim = out_dim < sm_dim ? out_dim : sm_dim;

    for (int i = 0; i < min_dim; i++)
        out[i] = g_linear * linear_out[i] + g_softmax * softmax_out[i];
    /* Zero-pad if dimensions mismatch */
    for (int i = min_dim; i < out_dim; i++)
        out[i] = 0;

    free(linear_out);
    free(softmax_out);
    return 0;
}
