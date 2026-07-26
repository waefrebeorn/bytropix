/*
 * wubu_safetensors_model.c -- safetensors -> bytropix weight layout (C11).
 *
 * Maps HF Transformers tensor names to the per-layer matrices the
 * bytropix forward pass consumes. Dequantizes F16/BF16/F32 on the fly
 * via safetensors_reader. Keeps the adapter's dims for sizing.
 */

#include "wubu_safetensors_model.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct wubu_st_model {
    st_ctx          *st;
    wubu_adapter_t  ad;
    int              d_model;
    int              d_ff;
    int              n_experts;
    int              kv_dim;       // derived from gqa_kv_heads * gqa_head_dim
};

static int layer_kv_dim(const wubu_adapter_t *a) {
    return a->gqa_kv_heads * a->gqa_head_dim;
}

wubu_st_model_t *wubu_st_open(const char *path, const wubu_adapter_t *adapter) {
    if (!path || !adapter || !adapter->ok) return NULL;
    st_ctx *st = st_open(path);
    if (!st) return NULL;
    wubu_st_model_t *m = (wubu_st_model_t *)calloc(1, sizeof(*m));
    if (!m) { st_close(st); return NULL; }
    m->st = st;
    m->ad = *adapter;
    m->d_model = adapter->d_model;
    m->d_ff = adapter->d_ff;
    m->n_experts = adapter->n_experts;
    m->kv_dim = layer_kv_dim(adapter);
    return m;
}

int wubu_st_n_layers(const wubu_st_model_t *m) { return m ? m->ad.n_layers : 0; }

static int fetch(const wubu_st_model_t *m, const char *name, float *out, int64_t n) {
    const st_tensor_info *t = st_find_tensor(m->st, name);
    if (!t) return 0;
    return st_read_tensor_f32(m->st, t, out, n) == n ? 1 : 0;
}

int wubu_st_layer_attn(const wubu_st_model_t *m, int layer,
                          float *q, float *k, float *v, float *o) {
    if (!m) return 0;
    char nq[128], nk[128], nv[128], no[128];
    snprintf(nq, sizeof(nq), "model.layers.%d.self_attn.q_proj.weight", layer);
    snprintf(nk, sizeof(nk), "model.layers.%d.self_attn.k_proj.weight", layer);
    snprintf(nv, sizeof(nv), "model.layers.%d.self_attn.v_proj.weight", layer);
    snprintf(no, sizeof(no), "model.layers.%d.self_attn.o_proj.weight", layer);
    int ok = 1;
    ok &= fetch(m, nq, q, (int64_t)m->d_model * m->d_model);
    ok &= fetch(m, nk, k, (int64_t)m->d_model * m->kv_dim);
    ok &= fetch(m, nv, v, (int64_t)m->d_model * m->kv_dim);
    ok &= fetch(m, no, o, (int64_t)m->d_model * m->d_model);
    return ok ? 1 : 0;
}

int wubu_st_layer_mlp(const wubu_st_model_t *m, int layer,
                         float *gate, float *up, float *down) {
    if (!m) return 0;
    char ng[128], nu[128], nd[128];
    snprintf(ng, sizeof(ng), "model.layers.%d.mlp.gate_proj.weight", layer);
    snprintf(nu, sizeof(nu), "model.layers.%d.mlp.up_proj.weight", layer);
    snprintf(nd, sizeof(nd), "model.layers.%d.mlp.down_proj.weight", layer);
    int ok = 1;
    ok &= fetch(m, ng, gate, (int64_t)m->d_model * m->d_ff);
    ok &= fetch(m, nu, up,   (int64_t)m->d_model * m->d_ff);
    ok &= fetch(m, nd, down, (int64_t)m->d_ff * m->d_model);
    return ok ? 1 : 0;
}

int wubu_st_layer_moe(const wubu_st_model_t *m, int layer,
                         float *router, int expert_idx,
                         float *egate, float *eup, float *edown) {
    if (!m || expert_idx < 0 || expert_idx >= m->n_experts) return 0;
    char nr[128], ng[160], nu[160], nd[160];
    snprintf(nr, sizeof(nr), "model.layers.%d.mlp.gate.weight", layer);
    snprintf(ng, sizeof(ng), "model.layers.%d.mlp.experts.gate_proj.weight", layer);
    snprintf(nu, sizeof(nu), "model.layers.%d.mlp.experts.up_proj.weight", layer);
    snprintf(nd, sizeof(nd), "model.layers.%d.mlp.experts.down_proj.weight", layer);
    int ok = 1;
    ok &= fetch(m, nr, router, (int64_t)m->d_model * m->n_experts);
    // Per-expert slices: weight[name][expert_idx] is [d_model, d_ff] (gate/up)
    // or [d_ff, d_model] (down). We read the full tensor and extract the slice.
    const st_tensor_info *t = st_find_tensor(m->st, ng);
    if (!t) return 0;
    int64_t per = (int64_t)m->d_model * m->d_ff;
    float *full = (float *)malloc((size_t)per * (size_t)m->n_experts * sizeof(float));
    if (!full) return 0;
    st_read_tensor_f32(m->st, t, full, per * m->n_experts);
    memcpy(egate, full + (size_t)expert_idx * per, (size_t)per * sizeof(float));
    free(full);

    t = st_find_tensor(m->st, nu);
    if (!t) return 0;
    full = (float *)realloc(full, (size_t)per * (size_t)m->n_experts * sizeof(float));
    st_read_tensor_f32(m->st, t, full, per * m->n_experts);
    memcpy(eup, full + (size_t)expert_idx * per, (size_t)per * sizeof(float));

    t = st_find_tensor(m->st, nd);
    int64_t per_d = (int64_t)m->d_ff * m->d_model;
    float *fulld = (float *)malloc((size_t)per_d * (size_t)m->n_experts * sizeof(float));
    st_read_tensor_f32(m->st, t, fulld, per_d * m->n_experts);
    memcpy(edown, fulld + (size_t)expert_idx * per_d, (size_t)per_d * sizeof(float));
    free(full); free(fulld);
    return ok ? 1 : 0;
}

int wubu_st_embed(const wubu_st_model_t *m, float *embd) {
    if (!m) return 0;
    int64_t n = (int64_t)m->ad.d_model;          // vocab unknown without header; caller sizes
    return fetch(m, "model.embed_tokens.weight", embd, n);   // partial safe
}

int wubu_st_lm_head(const wubu_st_model_t *m, float *lm_head) {
    if (!m) return 0;
    int64_t n = (int64_t)m->ad.d_model;
    return fetch(m, "lm_head.weight", lm_head, n);
}

void wubu_st_close(wubu_st_model_t *m) {
    if (!m) return;
    if (m->st) st_close(m->st);
    free(m);
}
