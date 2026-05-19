#ifndef WUBU_MODEL_H
#define WUBU_MODEL_H

#include "wubu_ssm.h"
#include "wubu_moe.h"
#include <stdbool.h>
#include <math.h>
#include <string.h>

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
#define GQA_MAX_CTX 262144  // max cached positions for KV cache (256k context)
#define GQA_KV_DIM (GQA_KV_HEADS * GQA_HEAD_DIM)  // 512

// KV cache format: 0=F32, 1=F16 (halves memory at cost of conversion)
#ifndef KV_CACHE_F16
#define KV_CACHE_F16 1  // default to F16 for memory efficiency
#endif

// F16 <-> F32 conversion helpers (used by KV cache)
static inline float fp16_to_fp32(uint16_t v) {
    int sign = (v >> 15) & 1;
    int exp  = (v >> 10) & 0x1F;
    int mant =  v        & 0x03FF;
    if (exp == 0) return ldexpf((float)mant / 1024.0f, -14) * (sign ? -1.0f : 1.0f);
    if (exp == 31) return sign ? -INFINITY : INFINITY;
    return ldexpf(1.0f + (float)mant / 1024.0f, exp - 15) * (sign ? -1.0f : 1.0f);
}
static inline uint16_t fp32_to_fp16(float v) {
    uint32_t bits; memcpy(&bits, &v, 4);
    int sign = (bits >> 31) & 1;
    int exp  = (bits >> 23) & 0xFF;
    int mant = bits & 0x7FFFFF;
    uint16_t fp16;
    if (exp == 0) { fp16 = (sign << 15) | (0) | (mant >> 13); }
    else if (exp == 0xFF) { fp16 = (sign << 15) | (31 << 10) | (mant >> 13); }
    else {
        int newexp = exp - 127 + 15;
        if (newexp >= 31) fp16 = (sign << 15) | (31 << 10);
        else if (newexp <= 0) fp16 = (sign << 15);
        else fp16 = (sign << 15) | (newexp << 10) | (mant >> 13);
    }
    return fp16;
}

// KV cache access helpers
static inline float kv_cache_read_elem(const void *cache, int64_t idx) {
#if KV_CACHE_F16
    return fp16_to_fp32(((const uint16_t *)cache)[idx]);
#else
    return ((const float *)cache)[idx];
#endif
}
// KV cache write: store float value into cache
static inline void kv_cache_write_elem(void *cache, int64_t idx, float val) {
#if KV_CACHE_F16
    ((uint16_t *)cache)[idx] = fp32_to_fp16(val);
#else
    ((float *)cache)[idx] = val;
#endif
}
// Batch read one head (256 floats) from F16 cache into float buffer
static inline void kv_cache_read_head(const void *cache, int64_t offset,
                                       float *buf, int n) {
#if KV_CACHE_F16
    const uint16_t *src = (const uint16_t *)cache + offset;
    for (int i = 0; i < n; i++) buf[i] = fp16_to_fp32(src[i]);
#else
    memcpy(buf, (const float *)cache + offset, n * sizeof(float));
#endif
}
// Batch write one head to F16 cache from float buffer
static inline void kv_cache_write_head(void *cache, int64_t offset,
                                        const float *buf, int n) {
#if KV_CACHE_F16
    uint16_t *dst = (uint16_t *)cache + offset;
    for (int i = 0; i < n; i++) dst[i] = fp32_to_fp16(buf[i]);
#else
    memcpy((float *)cache + offset, buf, n * sizeof(float));
#endif
}
// KV cache allocation: returns number of bytes needed for n_elems
static inline int64_t kv_cache_alloc_size(int64_t n_elems) {
#if KV_CACHE_F16
    return n_elems * (int64_t)sizeof(uint16_t);
#else
    return n_elems * (int64_t)sizeof(float);
#endif
}

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
    
    // GQA KV cache (10 GQA layers, max 256k context)
    void *gqa_k_cache;  // [10 * GQA_MAX_CTX * GQA_KV_DIM] F32 or F16
    void *gqa_v_cache;  // [10 * GQA_MAX_CTX * GQA_KV_DIM]
    int gqa_cache_len;   // how many tokens cached per layer (all 10 layers same len)
    
    // GGUF context (for per-layer MoE lazy loading)
    // Model state save/restore (for speculative decode rollback)
    float *ssm_states_saved;    // same size as ssm_states
    float *conv_states_saved;   // same size as conv_states
    int gqa_cache_len_saved;
    int mtp_cache_len_saved;

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

    // GPU acceleration context (opaque pointer, managed by wubu_model_gpu.cu)
    // When non-NULL, GQA layers run on GPU via chunked attention.
    void *gpu_ctx;
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

// ================================================================
// GPU-Accelerated Forward Path (wubu_model_gpu.cu)
// ================================================================

// Initialize GPU context: upload GQA weights, allocate KV cache + scratch.
// max_ctx: maximum KV cache positions (e.g. 262144).
// chunk_sz: max tokens per GPU batch (e.g. 512).
// Returns 1 on success, 0 on failure.
// When GPU context is active, wubu_model_forward() automatically uses
// GPU for GQA attention layers.
int wubu_model_gpu_init(wubu_model_t *model, int max_ctx, int chunk_sz);

// Run one GQA layer on GPU.
// Internal: called by wubu_model_forward when gpu_ctx != NULL.
int wubu_model_gpu_gqa_forward(wubu_model_t *model, int layer_idx,
                                const float *h_norm, int C, float *h_attn);

// Run SSM projections (qkv, gate) on GPU via quantized matmul kernels.
// h_norm: [C, D_MODEL] input
// C: number of tokens (1 for decode)
// qkv_out: [C, CONV_DIM] output (host)
// z_out: [C, VALUE_DIM] output (host)
// ssm_out_out: unused (future: ssm output projection)
int wubu_model_gpu_ssm_project(wubu_model_t *model, int layer_idx,
                                const float *h_norm, int C,
                                float *qkv_out, float *z_out,
                                float *ssm_out_out);

// Free all GPU resources and reset gpu_ctx to NULL.
void wubu_model_gpu_free(wubu_model_t *model);

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

// Model state save/restore for speculative decode rollback
bool wubu_model_checkpoint(wubu_model_t *model);
void wubu_model_rollback(wubu_model_t *model);

#ifdef __cplusplus
}
#endif

#endif // WUBU_MODEL_H
