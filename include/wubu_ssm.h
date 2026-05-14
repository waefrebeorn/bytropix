#ifndef WUBU_SSM_H
#define WUBU_SSM_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Qwen3.6-35B-A3B Gated Delta Net (SSM) Module
// ============================================================

// Hyperparameters (fixed for Qwen3.6-35B-A3B qwen35moe architecture)
#define D_MODEL     2048   // hidden dimension
#define D_INNER     4096   // SSM inner dimension (value_dim)
#define SSM_K_HEADS 16     // SSM num_k_heads (ssm_n_group)
#define SSM_V_HEADS 32     // SSM num_v_heads (ssm_dt_rank)
#define SSM_D_STATE 128    // SSM state dimension (head_k_dim = head_v_dim)
#define KEY_DIM     (SSM_D_STATE * SSM_K_HEADS)   // 2048
#define VALUE_DIM   (SSM_D_STATE * SSM_V_HEADS)   // 4096
#define CONV_DIM    (KEY_DIM * 2 + VALUE_DIM)     // 8192
#define CONV_KERNEL 4      // conv1d kernel size
#define DT_RANK     32     // ssm_time_step_rank

// GQA hyperparameters
#define GQA_Q_HEADS    16
#define GQA_KV_HEADS   2
#define GQA_HEAD_DIM   256

// All weights for one SSM layer
typedef struct {
    // Fused QKV projection: x @ attn_qkv -> [Q(2048), K(2048), V(4096)]
    float *attn_qkv_weight;  // [D_MODEL, KEY_DIM*2+VALUE_DIM] = [2048, 8192]
    
    // Gate (z) projection: x @ attn_gate -> [4096]
    float *attn_gate_weight;  // [D_MODEL, VALUE_DIM] = [2048, 4096]
    
    // SSM projections
    float *ssm_beta_weight;   // [D_MODEL, DT_RANK] = [2048, 32]
    float *ssm_alpha_weight;  // [D_MODEL, DT_RANK] = [2048, 32]
    float *ssm_dt_bias;       // [DT_RANK] = [32]
    float *ssm_a;             // [DT_RANK] = [32]  (-A_log)
    
    // Convolution
    float *ssm_conv1d_weight; // [CONV_KERNEL, CONV_DIM] = [4, 8192]
    
    // Gated normalization
    float *ssm_norm_weight;   // [SSM_D_STATE] = [128]
    
    // Output projection
    float *ssm_out_weight;    // [VALUE_DIM, D_MODEL] = [4096, 2048]
    
    // Pre-attention and post-attention norms
    float *attn_norm_weight;          // [D_MODEL] = [2048]
    float *post_attention_norm_weight; // [D_MODEL] = [2048]
} ssm_layer_weights;

// All weights for one GQA layer
typedef struct {
    // Q + gate fused: wq [D_MODEL, GQA_Q_HEADS*GQA_HEAD_DIM*2] = [2048, 8192]
    float *attn_q_weight;      // [2048, 8192]
    // K projection
    float *attn_k_weight;      // [D_MODEL, GQA_KV_HEADS*GQA_HEAD_DIM] = [2048, 512]
    // V projection
    float *attn_v_weight;      // [D_MODEL, GQA_KV_HEADS*GQA_HEAD_DIM] = [2048, 512]
    // Output projection
    float *attn_output_weight; // [GQA_Q_HEADS*GQA_HEAD_DIM, D_MODEL] = [4096, 2048]
    
    // Q/K norms
    float *attn_q_norm_weight;  // [GQA_HEAD_DIM] = [256]
    float *attn_k_norm_weight;  // [GQA_HEAD_DIM] = [256]
    
    // Pre/post norms
    float *attn_norm_weight;          // [D_MODEL] = [2048]
    float *post_attention_norm_weight; // [D_MODEL] = [2048]
} gqa_layer_weights;

// Full model state (for SSM recurrent state)
typedef struct {
    int n_layers;
    bool *is_ssm;             // layer_types[40]: which layers are SSM
    
    // Per-layer weights (union of SSM and GQA)
    ssm_layer_weights *ssm_layers;   // 30 layers
    gqa_layer_weights *gqa_layers;   // 10 layers
    
    // SSM recurrent states [layer][head][128][128]
    float ***ssm_states;  // [n_layers][SSM_V_HEADS][SSM_D_STATE][SSM_D_STATE]
    
    // Conv states [layer][conv_kernel-1][conv_dim]
    float **conv_states;  // [n_layers][CONV_KERNEL-1][CONV_DIM]
} wubu_model;

// ============================================================
// Forward pass functions
// ============================================================

// Single SSM layer forward pass
// x: [B, T, D_MODEL]
// weights: SSM layer weights
// ssm_state: [SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE] (mutable)
// conv_state: [CONV_KERNEL-1, CONV_DIM] (mutable)
// output: [B, T, D_MODEL]
void wubu_ssm_forward(const float *x, int B, int T,
                      const ssm_layer_weights *weights,
                      float *ssm_state,
                      float *conv_state,
                      float *output);

// Single GQA layer forward pass
// x: [B, T, D_MODEL]
// weights: GQA layer weights
// output: [B, T, D_MODEL]
void wubu_gqa_forward(const float *x, int B, int T,
                      const gqa_layer_weights *weights,
                      float *output);

// Single Poincaré SSM layer forward pass (hyperbolic recurrence)
// Same interface as wubu_ssm_forward but uses Möbius operations
// for the recurrence step
void wubu_poincare_ssm_forward(const float *x, int B, int T,
                               const ssm_layer_weights *weights,
                               float *ssm_state,
                               float *conv_state,
                               float R,
                               float *output);

// Single Poincaré GQA forward pass (hyperbolic attention)
// Same interface as wubu_gqa_forward but uses Poincaré distance
// instead of dot-product attention.
void wubu_poincare_gqa_forward(const float *x, int B, int T,
                               const gqa_layer_weights *weights,
                               float R,
                               float *output);

// Utility functions
int wubu_is_ssm_layer(int layer_idx);
void wubu_softplus(int n, const float *x, float *out);
void wubu_silu(int n, const float *x, float *out);
void wubu_sigmoid(int n, const float *x, float *out);
void wubu_l2_norm(int B, int T, int n_heads, int d,
                 const float *x, float eps, float *out);
void wubu_rms_norm(int B, int T, int d,
                   const float *x, const float *weight, float eps, float *out);
void wubu_conv1d(int B, int T, int C, int k,
                 const float *input, const float *kernel,
                 float *output);

// Qwen3.6 MRoPE
void wubu_rope(int B, int T, int n_heads, int head_dim,
               const float *x, const int *positions,
               int n_rot, const int *sections,
               float base, float *output);

#ifdef __cplusplus
}
#endif

#endif // WUBU_SSM_H
