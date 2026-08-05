#ifndef WUBU_MODEL_ADAPTER_H
#define WUBU_MODEL_ADAPTER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * wubu_model_adapter.h -- map a HuggingFace Transformers config.json
 * to wubuwizard's model dimensions + detect the architecture + LoRA.
 *
 * Covers the four new Colonel models:
 *   - Qwen3.6-27B        : dense Qwen3.6 hybrid (Gated DeltaNet SSM + GQA)
 *   - Agents-A1-4B        : dense Qwen-family (qwen3 parser)
 *   - KAT-Coder-V2.5-Dev : MoE on Qwen3.6-35B-A3B
 *   - BTL-3               : rank-32 LoRA adapter on Qwen3.6-27B base
 *
 * The adapter is the bridge between HuggingFace's "architectures" /
 * "model_type" strings and wubuwizard's internal d_model / n_experts /
 * tensor_naming. Opaque: caller only reads the resolved struct.
 */

typedef enum {
    WUBU_ARCH_UNKNOWN = 0,
    WUBU_ARCH_QWEN36_HYBRID,   // dense Qwen3.6 (SSM+GQA)
    WUBU_ARCH_QWEN_FAMILY,      // generic Qwen-family dense
    WUBU_ARCH_KAT_MOE,          // KAT-Coder MoE on Qwen3.6
    WUBU_ARCH_BTL3_LORA,        // BTL-3 LoRA-on-Qwen3.6-27B
    WUBU_ARCH_DEEPSEEK_V4_MOE   // DeepSeek-V4-Flash: 284B MXFP4 MoE + MLA
} wubu_arch_t;

typedef struct {
    wubu_arch_t arch;
    bool  is_moe;
    bool  is_lora;          // true for BTL-3 (needs base + adapter)
    char  base_model[256];  // for LoRA: the base checkpoint id
    int   lora_r;           // LoRA rank (BTL-3 = 32)
    int   lora_alpha;       // LoRA alpha (BTL-3 = 64)
    int   d_model;
    int   d_ff;             // expert / FFN intermediate dim
    int   n_experts;        // total routed experts (MoE)
    int   n_active_experts;  // top-k (MoE)
    int   n_layers;
    int   gqa_q_heads;
    int   gqa_kv_heads;
    int   gqa_head_dim;
    float rope_theta;
    float partial_rotary_factor;
    int   ssm_v_heads;       // linear_num_value_heads (KAT=32, Qwen27B=48)
    int   ssm_k_heads;       // linear_num_key_heads (16)
    int   ssm_value_head_dim;// linear_value_head_dim (128)
    int   ssm_conv_kernel;   // linear_conv_kernel_dim (4)
    int   ssm_d_state;       // SSM_D_STATE (128)
    int   shared_expert_ff;  // shared_expert_intermediate_size (512)
    int   full_attention_interval; // hybrid: every Nth layer is full_attention
    bool  attn_output_gate;  // attn_output_gate (true for both)
    bool  is_hybrid;         // layer_types mixes linear_attention + full_attention
    int   layer_types[256];  // 0=linear_attention(SSM+GQA), 1=full_attention(GQA only)
    int   vocab_size;        // from vocab_size (fallback 248320)
    int   tensor_naming;     // 0=blk.Qwen 1=model.layers.Gemma 2=pure-GQA
    bool  ok;
} wubu_adapter_t;

// Parse a HF config.json file path. Returns true on success (out->ok=true).
bool wubu_adapter_load(wubu_adapter_t *out, const char *config_json_path);

// Resolve a model by HF repo id / directory name (heuristic, no network).
// Used when only the model name is known. Returns true if a known mapping matched.
bool wubu_adapter_resolve_name(wubu_adapter_t *out, const char *name_or_id);

// Human-readable arch name (static buffer, valid until next call).
const char *wubu_arch_name(wubu_arch_t a);

#ifdef __cplusplus
}
#endif

#endif // WUBU_MODEL_ADAPTER_H
