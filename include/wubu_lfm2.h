#ifndef WUBU_LFM2_H
#define WUBU_LFM2_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * LFM2.5 (Liquid AI) hybrid loader + forward -- self-contained,
 * C11, opaque structs. Reuses wubuwizard's quantized_matmul /
 * rmsnorm helpers but owns its own gated-conv + GQA dispatch.
 *
 * Conv block (Liquid LFM2 technical report, arxiv 2511.23404):
 *   (B, C, h_tilde) = in_proj(h)        // [3*conv_dim, d_model]
 *   y = B * h_tilde                    // input gate
 *   z = depthwise_causal_conv_k(y)     // kernel k (3 for LFM2.5)
 *   o = out_proj(C * z)                // output gate + linear
 * No recurrent state -- pure conv.
 * ============================================================ */

typedef struct {
    /* Conv (SSM) block -- all F32 (dequantized from BF16) */
    float *in_proj;     /* [3*conv_dim, d_model] */
    float *conv_w;      /* [conv_dim, k] depthwise causal conv weights */
    float *out_proj;    /* [d_model, conv_dim] */
    int    conv_k;      /* conv kernel (3) */

    /* GQA attention block */
    float *q_proj;      /* [d_model, d_model] (n_q*hd) */
    float *k_proj;      /* [kv_dim, d_model]   (n_kv*hd) */
    float *v_proj;      /* [kv_dim, d_model] */
    float *o_proj;      /* [d_model, d_model] */
    float *q_ln;        /* [hd] layernorm gamma */
    float *k_ln;        /* [hd] layernorm gamma */

    /* SwiGLU FFN */
    float *w1;          /* [ff, d_model] gate */
    float *w2;          /* [d_model, ff] down */
    float *w3;          /* [ff, d_model] up */

    /* Norms */
    float *ffn_norm;    /* [d_model] */
    float *op_norm;     /* [d_model] operator (attn/conv) norm */
} lfm2_layer_t;

typedef struct {
    int n_layers;
    int d_model;
    int conv_dim;
    int n_q_heads;
    int n_kv_heads;
    int head_dim;
    int ff_dim;
    int vocab_size;
    float rope_theta;
    bool *is_conv;      /* per-layer: true=conv block, false=GQA */
    lfm2_layer_t *layers;
    float *embed;       /* [vocab, d_model] (tied with lm_head) */
    float *embed_norm;  /* [d_model] */
    /* KV cache for attention layers: [n_layers][2][n_kv_heads*head_dim*maxT] */
    float *kv_cache;
    int    kv_max_t;
} lfm2_model_t;

/* Load a LFM2.5 safetensors checkpoint directory into an lfm2_model_t.
 * Returns true on success. Self-contained: maps model.layers.N.* names. */
bool lfm2_load(const char *model_dir, lfm2_model_t *m);

/* Free all owned buffers. */
void lfm2_free(lfm2_model_t *m);

/* Forward one sequence of token embeddings. emb[B*T*d_model] in,
 * logits[vocab] out (last token). Allocates scratch internally. */
bool lfm2_forward(const lfm2_model_t *m, const float *emb, int B, int T,
                  float *logits);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_LFM2_H */
