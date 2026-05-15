#ifndef WUBU_POINCARE_GQA_H
#define WUBU_POINCARE_GQA_H

/**
 * Poincaré GQA (hyperbolic distance attention) forward + backward.
 *
 * Replaces standard softmax dot-product attention with hyperbolic
 * distance-based attention in the Poincaré ball.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "wubu_ssm.h"

// ============================================================
// Hyperbolic KV Cache for autoregressive generation
// ============================================================
// Stores K_ball and V_ball (post-exp_map) so subsequent calls
// don't recompute exp_map for past tokens. Append-only.
typedef struct {
    float *K_ball_cached;  // [max_T * GQA_KV_HEADS * GQA_HEAD_DIM]
    float *V_ball_cached;  // [max_T * GQA_KV_HEADS * GQA_HEAD_DIM]
    int max_T;             // allocated capacity in timesteps
    int current_T;         // number of timesteps currently stored
} poincare_kv_cache_t;

// Initialize cache with given capacity (in timesteps)
void poincare_kv_cache_init(poincare_kv_cache_t *cache, int init_capacity);

// Grow cache if needed (preserves existing data)
void poincare_kv_cache_resize(poincare_kv_cache_t *cache, int new_capacity);

// Free cache memory
void poincare_kv_cache_free(poincare_kv_cache_t *cache);

// ============================================================
// Saved intermediates for backward pass
// ============================================================
typedef struct {
    float *Q_norm;              // [N, q_dim] post-RMSNorm Q
    float *Q_raw;               // [N, q_dim] pre-RMSNorm Q
    float *K_norm;              // [N, kv_dim] post-RMSNorm K
    float *K_raw;               // [N, kv_dim] pre-RMSNorm K
    float *V;                   // [N, kv_dim] raw V (pre-exp_map)
    float *gate;                // [N, q_dim] raw gate (pre-sigmoid)
    float *gate_sig;            // [N, q_dim] sigmoid(gate)
    float *attn_out_pre_gate;   // [N, q_dim] attn_out before gating
    float *Q_ball;              // [N, q_dim] exp_map(Q_norm)
    float *K_ball;              // [N, kv_dim] exp_map(K_norm) — new tokens only when cache used
    float *V_ball;              // [N, kv_dim] exp_map(V) — new tokens only when cache used
    poincare_kv_cache_t *cache; // optional non-NULL to enable KV cache
} poincare_gqa_fwd_save_t;

// ============================================================
// Poincaré GQA Forward Pass
// ============================================================
void wubu_poincare_gqa_forward(const float *x, int B, int T,
                               const gqa_layer_weights *weights,
                               float R,
                               float *output);

// ============================================================
// Poincaré GQA Forward + Save Intermediates
// ============================================================
void wubu_poincare_gqa_forward_save(const float *x, int B, int T,
                                    const gqa_layer_weights *weights,
                                    float R,
                                    float *output,
                                    poincare_gqa_fwd_save_t *save);

// ============================================================
// Poincaré GQA Backward Pass (hyperbolic distance attention)
// ============================================================
void wubu_poincare_gqa_backward(
    int B, int T,
    const float *x,               // [B, T, D_MODEL] forward input
    const float *Q_norm,          // [N, q_dim] post-RMSNorm Q
    const float *Q_raw,           // [N, q_dim] pre-RMSNorm Q
    const float *K_norm,          // [N, kv_dim] post-RMSNorm K
    const float *K_raw,           // [N, kv_dim] pre-RMSNorm K
    const float *V,               // [N, kv_dim] raw V
    const float *Q_ball,          // [N, q_dim] Q in ball
    const float *K_ball,          // [N, kv_dim] K in ball
    const float *V_ball,          // [N, kv_dim] V in ball
    const float *gate,            // [N, q_dim] pre-sigmoid gate
    const float *gate_sig,        // [N, q_dim] sigmoid(gate)
    const float *attn_out,        // [N, q_dim] post-gate attn_out
    const float *output,          // [B, T, D_MODEL] forward output
    const float *d_output,        // [B, T, D_MODEL] upstream grad
    const gqa_layer_weights *w,
    float R,                      // Poincaré ball radius
    float *d_x,                   // [B, T, D_MODEL] output grad
    float *d_q_weight,            // [D_MODEL, q_dim*2]
    float *d_k_weight,            // [D_MODEL, kv_dim]
    float *d_v_weight,            // [D_MODEL, kv_dim]
    float *d_q_norm_weight,       // [GQA_HEAD_DIM]
    float *d_k_norm_weight,       // [GQA_HEAD_DIM]
    float *d_out_weight);         // [q_dim, D_MODEL]

#ifdef __cplusplus
}
#endif

#endif // WUBU_POINCARE_GQA_H
