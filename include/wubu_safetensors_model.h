#ifndef WUBU_SAFETENSORS_MODEL_H
#define WUBU_SAFETENSORS_MODEL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "safetensors_reader.h"
#include "wubu_model_adapter.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * wubu_safetensors_model.h -- load a HuggingFace safetensors checkpoint
 * into wubuwizard's internal SSM+GQA+MoE weight layout.
 *
 * wubuwizard's native path reads GGUF (blk.N.* names). The Colonel models
 * ship as safetensors with Transformers naming:
 *   model.layers.N.self_attn.q_proj.weight        [d_model, d_model]
 *   model.layers.N.self_attn.k_proj.weight        [d_model, kv_dim]
 *   model.layers.N.self_attn.v_proj.weight        [d_model, kv_dim]
 *   model.layers.N.self_attn.o_proj.weight        [d_model, d_model]
 *   model.layers.N.mlp.gate_proj.weight          [d_model, d_ff]
 *   model.layers.N.mlp.up_proj.weight            [d_model, d_ff]
 *   model.layers.N.mlp.down_proj.weight          [d_ff, d_model]
 *   model.layers.N.mlp.experts.gate_proj.weight [n_experts, d_model, d_ff]
 *   model.layers.N.mlp.experts.up_proj.weight   [n_experts, d_model, d_ff]
 *   model.layers.N.mlp.experts.down_proj.weight [n_experts, d_ff, d_model]
 *   model.layers.N.mlp.gate.weight (router)      [d_model, n_experts]
 *   model.norm.weight / model.layers.N.input_layernorm.weight
 *   model.embed_tokens.weight                    [vocab, d_model]
 *   lm_head.weight                              [vocab, d_model]
 *
 * The SSM (Gated DeltaNet) projection is the fused qkv in wubuwizard's
 * internal layout; for the Transformers models we map the attention
 * projections directly to gqa_layer_weights and the MLP to moe/expert.
 *
 * This header is intentionally minimal and self-contained; the heavy
 * numeric code lives in wubu_safetensors_model.c.
 */

typedef struct wubu_st_model wubu_st_model_t;

// Open a safetensors checkpoint + a parsed adapter (config). Returns NULL on fail.
wubu_st_model_t *wubu_st_open(const char *safetensors_path,
                                const wubu_adapter_t *adapter);

// Number of layers detected.
int wubu_st_n_layers(const wubu_st_model_t *m);

// Fetch a layer's Q/K/V/O projection as dequantized F32 row-major matrices.
// Each returns 1 on success (and writes into caller-provided buffers of
// the correct size) or 0 if absent. Sizes:
//   q: [d_model, d_model];  k/v: [d_model, kv_dim];  o: [d_model, d_model]
int wubu_st_layer_attn(const wubu_st_model_t *m, int layer,
                          float *q, float *k, float *v, float *o);

// Fetch a layer's MLP (gate/up/down) as F32. Sizes:
//   gate/up: [d_model, d_ff];  down: [d_ff, d_model]
int wubu_st_layer_mlp(const wubu_st_model_t *m, int layer,
                         float *gate, float *up, float *down);

// Fetch router (gate) weight [d_model, n_experts] + one expert's
// gate/up/down for a MoE layer. expert idx in [0, n_experts).
int wubu_st_layer_moe(const wubu_st_model_t *m, int layer,
                         float *router,
                         int expert_idx, float *egate, float *eup, float *edown);

// Fetch embedding / lm_head [vocab, d_model] (caller sizes accordingly).
int wubu_st_embed(const wubu_st_model_t *m, float *embd);
int wubu_st_lm_head(const wubu_st_model_t *m, float *lm_head);

// Close + free.
void wubu_st_close(wubu_st_model_t *m);

#ifdef __cplusplus
}
#endif

#endif // WUBU_SAFETENSORS_MODEL_H
