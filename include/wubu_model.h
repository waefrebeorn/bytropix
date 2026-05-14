#ifndef WUBU_MODEL_H
#define WUBU_MODEL_H

#include "wubu_ssm.h"
#include "wubu_moe.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Layer configuration
typedef struct {
    int layer_idx;
    bool is_ssm;  // false = GQA
    
    // Weights (loaded from GGUF)
    ssm_layer_weights ssm;      // valid if is_ssm
    gqa_layer_weights gqa;      // valid if !is_ssm
    
    // Layer norm (pre-attention for all layers)
    float *attn_norm_weight;    // [D_MODEL], RMSNorm
    
    // Post-attention norm
    float *post_attn_norm_weight; // [D_MODEL], RMSNorm
    
    // MoE (FFN) weights
    moe_weights_t moe;
} wubu_layer_t;

// Complete model
typedef struct {
    int n_layers;
    wubu_layer_t *layers;
    
    // Token embedding
    float *token_embd;       // [vocab_size, D_MODEL] or NULL if using embedding file
    float *output_weight;    // [D_MODEL, vocab_size] or NULL
    
    // Embedding file (from Phase 1)
    bool use_embedding_file;
    int vocab_size;
    
    // Norms
    float *norm_weight;  // final RMSNorm [D_MODEL]
    
    // State buffers (reused across calls)
    float *ssm_states;    // [max_layers, SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE]
    float *conv_states;   // [max_layers, B, CONV_KERNEL-1, CONV_DIM]
    
    // GGUF context (for per-layer MoE lazy loading)
    struct gguf_ctx *gguf_ctx;
    
    // Enable MoE during forward (default: false for memory reasons)
    bool enable_moe;
    
    // MoE test: only load MoE for first N layers (0 = all)
    int moe_max_layers;
} wubu_model_t;

// Create model, load from GGUF
bool wubu_model_init(wubu_model_t *model, const char *gguf_path);

// Free model resources
void wubu_model_free(wubu_model_t *model);

// Forward pass through all layers
// Input: token_ids [B, T], Output: logits [B, T, vocab_size]
void wubu_model_forward(wubu_model_t *model,
                        const int *token_ids, int B, int T,
                        float *logits);

// Forward pass from embeddings (bypass token lookup)
// Input: embeddings [B, T, D_MODEL], Output: logits [B, T, vocab_size]
void wubu_model_forward_from_embd(wubu_model_t *model,
                                  const float *embeddings, int B, int T,
                                  float *logits);

// Model-level backward pass
// Requires saved layer outputs from forward (normed, attn_out, normed2, ffn_out arrays)
// All arrays are [n_layers * B * T * D_MODEL] flattened
// SSM/GQA intermediates arrays (per layer) — see wubu_ssm_backward / wubu_gqa_backward
// For MoE: gradient passes through (identity backward)
// Output: d_embeddings [B, T, D_MODEL]
void wubu_model_backward_from_embd(
    const wubu_model_t *model,
    const float *embeddings,
    const float *logits, const float *d_logits,
    const float *saved_normed,     // [n_layers * N * D_MODEL]
    const float *saved_attn_out,   // [n_layers * N * D_MODEL]
    const float *saved_normed2,    // [n_layers * N * D_MODEL]
    const float *saved_ffn_out,    // [n_layers * N * D_MODEL]
    float *d_embeddings,
    int B, int T);

#ifdef __cplusplus
}
#endif

#endif // WUBU_MODEL_H
