#ifndef WUBU_MODEL_SAFETENSORS_BRIDGE_H
#define WUBU_MODEL_SAFETENSORS_BRIDGE_H

#include "wubu_model.h"
#include "wubu_model_adapter.h"
#include "safetensors_reader.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * wubu_model_safetensors_bridge.h -- load a HuggingFace safetensors
 * checkpoint (the Colonel models: Qwen3.6-27B hybrid, Agents-A1-4B,
 * KAT-Coder MoE, BTL-3 LoRA) into bytropix's wubu_model_t and run it
 * through the EXISTING SSM + GQA + MoE forward passes in pure F32.
 *
 * No third-party deps, no GGUF assumption. Tensor names are the real
 * published HF names (model.language_model.layers.N.linear_attn.* etc.)
 * discovered from each repo's model.safetensors.index.json.
 *
 * Layout mapping (HF -> bytropix, all F32, transposed where needed):
 *   self_attn.q_proj/k_proj/v_proj/o_proj  -> gqa.attn_{q,k,v,output}_weight
 *   linear_attn.in_proj_qkv                 -> ssm.attn_qkv_weight_f32 [D,CONV_DIM]
 *   linear_attn.in_proj_z                   -> ssm.attn_gate_weight_f32 [D,VALUE_DIM]
 *   linear_attn.in_proj_a/b                 -> ssm.ssm_alpha/beta_weight [D,DT_RANK]
 *   linear_attn.A_log                       -> ssm.ssm_a [DT_RANK]
 *   linear_attn.dt_bias                     -> ssm.ssm_dt_bias [DT_RANK]
 *   linear_attn.convNd                      -> ssm.ssm_conv1d_weight [CONV_K,CONV_DIM]
 *   linear_attn.norm                       -> ssm.ssm_norm_weight [SSM_D_STATE]
 *   linear_attn.out_proj                   -> ssm.ssm_out_weight_f32 [VALUE_DIM,D]
 *   mlp.gate/up/down_proj                  -> moe single-expert (dense MLP)
 *   mlp.experts.N.{gate,up,down}_proj      -> moe.ffn_*_exps[:,:,N]  (transposed)
 *   mlp.gate / shared_expert_*              -> moe router / shared experts
 *   embed_tokens / lm_head                  -> token_embd / output_weight
 */

/* Allocate + populate a wubu_model_t from a safetensors checkpoint.
 * Returns 0 on success. On failure prints diagnostics and returns -1.
 * The caller must eventually call wubu_model_free(&m) (or a bridge free). */
int wubu_model_init_safetensors(wubu_model_t *m, const char *path,
                               const wubu_adapter_t *ad);

/* Like wubu_model_init_safetensors, but routes routed MoE experts through a
 * ds4-ssd sidecar (wubu_ssd_moe_t) instead of loading them resident.
 * sidecar_dir must contain experts.<L>.bin + manifest.json (see docs/ssd_moe.md).
 * Pass NULL for the standard in-RAM path. Returns 0 on success. */
int wubu_model_init_safetensors_ssd(wubu_model_t *m, const char *path,
                                    const wubu_adapter_t *ad,
                                    const char *sidecar_dir);

/* Convenience: open the checkpoint, detect arch via adapter, init. */
int wubu_model_init_auto(wubu_model_t *m, const char *path);

/* Free bridge-allocated F32 weight arrays inside m (those not owned by a
 * mmap'd blob). Safe to call before wubu_model_free. */
void wubu_model_safetensors_free(wubu_model_t *m);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_MODEL_SAFETENSORS_BRIDGE_H */
