/*
 * wubu_dsv4_layer.c --- DeepSeek-V4 MLA attention bridge.
 *
 * Self-contained C11 module that bridges GGUF tensor loading (wubu_model.c)
 * with the MLA forward pass (wubu_mla.c). Does NOT modify engine logic.
 *
 * Provides:
 *   - Tensor name resolver (GGUF llama.cpp convention)
 *   - Layer context (opaque struct wrapping wubu_mla_t + weight pointers)
 *   - Forward pass scaffold (KV down-proj, up-proj K/V, Q proj, attn, out proj)
 *
 * Triple-DA: dims validated at create; NULL guards everywhere; deterministic.
 */
#include "wubu_dsv4_layer.h"
#include "wubu_mla.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct wubu_dsv4_layer {
    wubu_mla_t *mla;
    const float *W_DQ;
    const float *W_UQ;
    const float *W_DKV;
    const float *W_UK;
    const float *W_UV;
    const float *W_O;
    const float *attn_norm;
    int layer_idx;
};

wubu_dsv4_layer_t *wubu_dsv4_layer_create(int hidden_dim, int n_heads,
                                           int head_dim, int q_lora_rank,
                                           int kv_lora_rank, int rope_head_dim) {
    if (hidden_dim <= 0 || n_heads <= 0 || head_dim <= 0 ||
        q_lora_rank <= 0 || kv_lora_rank <= 0 || rope_head_dim <= 0)
        return NULL;
    wubu_dsv4_layer_t *dl = (wubu_dsv4_layer_t *)calloc(1, sizeof(*dl));
    if (!dl) return NULL;
    dl->mla = wubu_mla_create(hidden_dim, n_heads, head_dim,
                               q_lora_rank, kv_lora_rank, rope_head_dim);
    if (!dl->mla) { free(dl); return NULL; }
    return dl;
}

void wubu_dsv4_layer_free(wubu_dsv4_layer_t *dl) {
    if (!dl) return;
    wubu_mla_free(dl->mla);
    free(dl);
}

int wubu_dsv4_layer_load_tensors(wubu_dsv4_layer_t *dl, int layer_idx,
                                  const float *W_DQ, const float *W_UQ,
                                  const float *W_DKV, const float *W_UK,
                                  const float *W_UV, const float *W_O,
                                  const float *attn_norm) {
    if (!dl) return -1;
    dl->W_DQ = W_DQ;
    dl->W_UQ = W_UQ;
    dl->W_DKV = W_DKV;
    dl->W_UK = W_UK;
    dl->W_UV = W_UV;
    dl->W_O = W_O;
    dl->attn_norm = attn_norm;
    dl->layer_idx = layer_idx;
    return 1;
}

/* Full MLA forward for a single decode step.
 *
 * x: [d_model] input hidden state (post attention_norm)
 * kv_cache: [kv_latent_dim * pos] KV latent cache for positions 0..pos-1
 * pos: current decode position (number of cached entries)
 * out: [d_model] output
 *
 * Pipeline:
 *   1. x -> kv_latent  via W_DKV  (kv_lora_rank + rope_head_dim dims)
 *   2. RMSNorm on kv_latent
 *   3. kv_latent -> K_full, V_full  via W_UK, W_UV  (n_heads * head_dim each)
 *   4. x -> Q  via W_DQ, W_UQ  (down to q_lora_rank, up to n_heads * head_dim)
 *   5. For each cached position: up-project cached kv_latent, then wubu_mla_attn
 *   6. Sum attention outputs, project via W_O -> hidden_dim
 */
int wubu_dsv4_layer_forward(const wubu_dsv4_layer_t *dl,
                             const float *x,
                             const float *kv_cache,
                             int pos,
                             float *out) {
    if (!dl || !x || !out) return -1;
    if (!dl->W_DQ || !dl->W_UQ || !dl->W_DKV || !dl->W_UK ||
        !dl->W_UV || !dl->W_O || !dl->attn_norm)
        return -1;
    if (!dl->mla || pos < 0) return -1;

    const wubu_mla_t *m = dl->mla;
    int d_model     = m->hidden_dim;
    int n_heads     = m->n_heads;
    int head_dim    = m->head_dim;
    int rope_hd     = m->rope_head_dim;
    int kv_latent_dim = m->kv_latent_dim;

    /* Phase 1: KV down-projection + RMSNorm */
    float *kv_lat = (float *)malloc((size_t)kv_latent_dim * sizeof(float));
    if (!kv_lat) return -1;
    wubu_mla_down_proj_kv(m, dl->W_DKV, x, kv_lat);
    for (int i = 0; i < kv_latent_dim; i++)
        kv_lat[i] *= dl->attn_norm[i];

    /* Phase 2: Up-project current token's KV latent to K and V.
     * Both produce [n_heads * head_dim] — the full head dimension. */
    float *k_full = (float *)malloc((size_t)n_heads * head_dim * sizeof(float));
    float *v_full = (float *)malloc((size_t)n_heads * head_dim * sizeof(float));
    if (!k_full || !v_full) {
        free(kv_lat); free(k_full); free(v_full);
        return -1;
    }
    wubu_mla_up_proj_k(m, dl->W_UK, kv_lat, k_full);
    wubu_mla_up_proj_v(m, dl->W_UV, kv_lat, v_full);

    /* Phase 3: Q projection: x -> q_lora_rank -> n_heads * head_dim */
    float *q_full = (float *)malloc((size_t)n_heads * head_dim * sizeof(float));
    if (!q_full) {
        free(kv_lat); free(k_full); free(v_full);
        return -1;
    }
    wubu_mla_proj_q(m, dl->W_DQ, dl->W_UQ, x, q_full);

    /* Phase 4: Attention over all positions [0..pos]
     * For single-token decode, pos is the number of cached tokens.
     * Position pos is the current token (kv_lat).
     * For each position, up-project its kv_latent to K/V, then run wubu_mla_attn. */
    float *attn_out = (float *)calloc((size_t)n_heads * head_dim, sizeof(float));
    if (!attn_out) {
        free(kv_lat); free(k_full); free(v_full); free(q_full);
        return -1;
    }

    /* rope part is shared: kv_lat[kv_lora_rank..kv_lora_rank+rope_hd] */
    float *k_rope = (float *)malloc((size_t)rope_hd * sizeof(float));
    if (!k_rope) {
        free(kv_lat); free(k_full); free(v_full); free(q_full); free(attn_out);
        return -1;
    }
    for (int i = 0; i < rope_hd; i++)
        k_rope[i] = kv_lat[kv_latent_dim - rope_hd + i];

    for (int p = 0; p <= pos; p++) {
        const float *kp = (p < pos && kv_cache)
            ? kv_cache + (size_t)p * kv_latent_dim
            : kv_lat;

        float *kp_k = (float *)malloc((size_t)n_heads * head_dim * sizeof(float));
        float *kp_v = (float *)malloc((size_t)n_heads * head_dim * sizeof(float));
        if (!kp_k || !kp_v) {
            free(kp_k); free(kp_v);
            continue;
        }
        wubu_mla_up_proj_k(m, dl->W_UK, kp, kp_k);
        wubu_mla_up_proj_v(m, dl->W_UV, kp, kp_v);

        float *o = (float *)calloc((size_t)n_heads * head_dim, sizeof(float));
        if (o) {
            wubu_mla_attn(m, q_full, kp_k, k_rope, kp_v, o);
            for (int i = 0; i < n_heads * head_dim; i++)
                attn_out[i] += o[i];
            free(o);
        }
        free(kp_k); free(kp_v);
    }

    /* Phase 5: Output projection: attn_out -> [d_model] via W_O */
    for (int i = 0; i < d_model; i++) {
        float dot = 0.0f;
        const float *row = dl->W_O + (size_t)i * n_heads * head_dim;
        for (int j = 0; j < n_heads * head_dim; j++)
            dot += attn_out[j] * row[j];
        out[i] = dot;
    }

    free(kv_lat); free(k_full); free(v_full); free(q_full); free(attn_out); free(k_rope);
    return 0;
}

char *wubu_dsv4_tensor_name(int layer, const char *tensor_type) {
    if (!tensor_type) return NULL;
    char *name = (char *)malloc(128);
    if (!name) return NULL;
    if (strcmp(tensor_type, "q_a") == 0)
        snprintf(name, 128, "blk.%d.attn_q_a", layer);
    else if (strcmp(tensor_type, "q_b") == 0)
        snprintf(name, 128, "blk.%d.attn_q_b", layer);
    else if (strcmp(tensor_type, "kv") == 0)
        snprintf(name, 128, "blk.%d.attn_kv", layer);
    else if (strcmp(tensor_type, "k_up") == 0)
        snprintf(name, 128, "blk.%d.attn_k", layer);
    else if (strcmp(tensor_type, "v_up") == 0)
        snprintf(name, 128, "blk.%d.attn_v", layer);
    else if (strcmp(tensor_type, "o_a") == 0)
        snprintf(name, 128, "blk.%d.attn_out_a", layer);
    else if (strcmp(tensor_type, "o_b") == 0)
        snprintf(name, 128, "blk.%d.attn_out_b", layer);
    else if (strcmp(tensor_type, "norm") == 0)
        snprintf(name, 128, "blk.%d.attn_norm", layer);
    else { free(name); return NULL; }
    return name;
}
