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
#define GQA_MAX_CTX 4096  // max cached positions for KV cache
#define GQA_KV_DIM (GQA_KV_HEADS * GQA_HEAD_DIM)  // 512

// MTP (Multi-Token Prediction) head for speculative decode
// Architecture: h_39 → hnorm → concat(hnorm, enorm(embd)) → eh_proj → blk.40 → shared_head_norm → output
typedef struct {
    bool loaded;
    
    // Nextn norms (F32, all [D_MODEL])
    float *nextn_hnorm;             // [2048] — hidden state norm
    float *nextn_enorm;             // [2048] — token embedding norm
    float *nextn_shared_head_norm;  // [2048] — output norm
    
    // eh_proj weight (F32 dequantized): concat([h_norm | e_norm], dim=4096) → [2048]
    float *nextn_eh_proj_f32;   // [4096, 2048] F32
    int64_t nextn_eh_proj_dim;         // 4096 (concat dim)
    
    // Blk.40 (a full GQA+MoE layer)
    wubu_layer_t blk40;
    
    // KV cache for blk.40's GQA attention
    float *k_cache;  // [GQA_MAX_CTX * GQA_KV_DIM]
    float *v_cache;  // [GQA_MAX_CTX * GQA_KV_DIM]
    int cache_len;
} mtp_head_t;
typedef struct {
    int n_layers;
    wubu_layer_t *layers;
    
    // Token embedding
    float *token_embd;       // [vocab_size, D_MODEL] or NULL if using embedding file
    float *output_weight;    // [D_MODEL, vocab_size] or NULL
    const uint8_t *output_weight_q;   // raw Q4_K quantized
    int output_weight_type;
    
    // Embedding file (from Phase 1)
    bool use_embedding_file;
    int vocab_size;
    
    // Norms
    float *norm_weight;  // final RMSNorm [D_MODEL]
    
    // State buffers (reused across calls)
    float *ssm_states;    // [max_layers, SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE]
    float *conv_states;   // [max_layers, B, CONV_KERNEL-1, CONV_DIM]
    
    // GQA KV cache (10 GQA layers, max 4096 context)
    float *gqa_k_cache;  // [10 * GQA_MAX_CTX * GQA_KV_DIM]
    float *gqa_v_cache;  // [10 * GQA_MAX_CTX * GQA_KV_DIM]
    int gqa_cache_len;   // how many tokens cached per layer (all 10 layers same len)
    
    // GGUF context (for per-layer MoE lazy loading)
    // GGUF context (for per-layer MoE lazy loading)
    gguf_ctx *gguf_ctx;
    
    // Enable MoE during forward (default: false for memory reasons)
    bool enable_moe;
    
    // Skip output projection in forward (for GPU offload)
    bool skip_output_proj;
    
    // MoE test: only load MoE for first N layers (0 = all)
    int moe_max_layers;
    
    // MTP (Multi-Token Prediction) head for speculative decode
    mtp_head_t mtp;
    
    // Last hidden state capture (for MTP: set to a [D_MODEL] buffer before forward)
    float *save_last_hidden;
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
// Backward pass from embeddings
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

// MTP: Load MTP head from a separate GGUF model file
// Must be called AFTER wubu_model_init on the main model
// Pass the MTP GGUF model path
bool wubu_mtp_load(mtp_head_t *mtp, const char *mtp_gguf_path,
                   gguf_ctx *main_ctx, const uint8_t *main_blob);

// MTP: Draft forward — predict next tokens from last hidden state
// x: [D_MODEL] — last hidden state from main model (layer 39 output, post-residual)
// token_embd: [B, D_MODEL] — embeddings of candidate continuation tokens
// B: number of draft candidates to evaluate
// logits_out: [B, vocab_size] — output logits for each candidate
// Returns: number of tokens consumed from token_embd (for KV cache tracking)
int wubu_mtp_draft_forward(wubu_model_t *model,
                           const float *x,
                           const float *token_embd, int B,
                           float *logits_out);

// Free MTP head resources
void wubu_mtp_free(mtp_head_t *mtp);

#ifdef __cplusplus
}
#endif

#endif // WUBU_MODEL_H
