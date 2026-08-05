/* wubu_lfm.h — LFM2.5-2.6B hybrid attention layer (C11, opaque, minimal)
 *
 * LFM2.5-2.6B (liquidai): hybrid architecture combining linear attention
 * (Gated DeltaNet) with standard softmax (GQA) attention in alternating
 * layers. 34T tokens, 128K context, 128K vocab, on-device optimized.
 *
 * Architecture:
 *   - Even layers (0,2,4,...): GLA (Gated Linear Attention / linear recurrent)
 *   - Odd layers  (1,3,5,...): standard GQA (grouped-query attention + softmax)
 *   - Hybrid combination: output = linear_attn(x) + softmax_attn(x)
 *     (both paths contribute, with a learned interpolation gate)
 *
 * Reuses:
 *   - wubu_linear_attn.h: wubu_deltanet_update (Gated DeltaNet),
 *     wubu_gla_update (GLA per-head gate), wubu_retnet_update (GSA retention)
 *   - Standard softmax attention computed inline (no external deps)
 *
 * Reference: liquidai LFM2.5-2.6B
 *   34T tokens trained, 128K context, designed for on-device inference.
 */
#ifndef WUBU_LFM_H
#define WUBU_LFM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque LFM2.5 layer handle */
typedef struct wubu_lfm wubu_lfm_t;

/* Configuration */
typedef struct {
    int d_model;       /* hidden dimension */
    int n_heads;       /* number of attention heads */
    int d_head;        /* per-head dimension */
    int n_kv_heads;    /* number of KV heads (GQA grouping) */
    int n_layers;      /* total layers */
    int hybrid_gate;   /* 1 = hybrid (linear + softmax), 0 = alternate only */
} wubu_lfm_cfg_t;

/* Create an LFM2.5 layer context. Returns NULL on bad args or OOM. */
wubu_lfm_t *wubu_lfm_create(const wubu_lfm_cfg_t *cfg);

/* Destroy context. NULL-safe. */
void wubu_lfm_free(wubu_lfm_t *lfm);

/* Linear attention path (Gated DeltaNet / GLA).
 * State S: [d_model * d_model] (DeltaNet state matrix, row-major)
 * k_lin, v_lin: [d_model] key/value for this step.
 * beta: decay factor.
 * Sout: [d_model * d_model] new state matrix (caller-allocated).
 * out:  [d_model] attention output (state @ key).
 * Reuses wubu_deltanet_update from wubu_linear_attn.c. */
int wubu_lfm_linear_attn(const float *S, const float *k_lin, const float *v_lin,
                          int d, float beta, float *Sout, float *out);

/* Standard softmax GQA attention (odd layers).
 * queries:  [n_heads * d_head]
 * keys:     [n_kv_heads * d_head * seq_len]  (KV-shared across head groups)
 * values:   [n_kv_heads * d_head * seq_len]
 * pos:      query position (for causal masking)
 * out:      [n_heads * d_head]
 * d_head:   per-head dimension
 * n_heads:  number of query heads
 * n_kv_heads: number of KV heads (n_heads must be divisible by n_kv_heads)
 * seq_len:  KV sequence length
 * Returns 0 on success, -1 on error. */
int wubu_lfm_softmax_attn(const float *queries,
                           const float *keys, const float *values,
                           int n_heads, int n_kv_heads, int d_head,
                           int seq_len, int pos, float *out);

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
 * Sout: [d_model * d_model] new linear state (caller-allocated)
 *
 * Returns 0 on success, -1 on error. */
int wubu_lfm_hybrid_layer(const wubu_lfm_t *lfm,
                           const float *linear_state,
                           const float *query, const float *key, const float *value,
                           const float *k_lin, const float *v_lin,
                           float gate, int layer_idx,
                           float *out, float *Sout);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_LFM_H */
