#ifndef WUBU_NESTED_SSM_H
#define WUBU_NESTED_SSM_H

#include "wubu_ssm.h"
#include "wubu_mobius.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Nested SSM: Product of K Poincaré balls with K curvatures.
 *
 * Generalizes the single-ball Poincaré SSM to K balls with different radii:
 *   h[t] = (h_1[t], ..., h_K[t])  where each h_k lives in a ball of radius R_k
 *
 * Each ball operates independently on the full state dimension:
 *   h_k[t] = mobius_add(scalar_mul(exp(gate), h_k[t-1]),
 *                        exp_map((k ⊗ (log_map(v) - log_map(h_k[t-1] @ k))) * beta, R_k))
 *
 * The gating mechanism uses the same beta/gate for all K balls, but a learned
 * per-ball weight w_k (softmax-normalized) controls each ball's contribution
 * to the final output.
 */

#define NESTED_SSM_MAX_K 16

/**
 * Nested SSM state: K independent Poincaré ball state matrices.
 *
 * Each ball has its own state matrix [SSM_V_HEADS, SSM_D_STATE, SSM_D_STATE]
 * and operates with its curvature radius R_k.
 *
 * Layout: states[k * HEAD_STATE_SIZE + (vh * SSM_D_STATE + i) * SSM_D_STATE + j]
 *   where HEAD_STATE_SIZE = SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE
 */
typedef struct {
    int K;          // Number of balls (1 <= K <= NESTED_SSM_MAX_K)
    float R[NESTED_SSM_MAX_K];  // Curvatures [K]
    float *states;  // [K * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE] — flat array
} wubu_nested_ssm_state_t;

/**
 * Nested SSM gating: per-ball learned weights for combining outputs.
 * Pass NULL for uniform weighting (1/K each).
 */
typedef struct {
    float ball_weights[NESTED_SSM_MAX_K];  // [K] — will be softmax-normalized at runtime
} wubu_nested_ssm_gating_t;

/**
 * Forward pass for nested Poincaré SSM (K balls, K curvatures).
 *
 * Same input/output interface as wubu_poincare_ssm_forward, but with
 * K independent Poincaré ball states combined via gating.
 *
 * @param x             Input tensor [B, T, D_MODEL]
 * @param B             Batch size
 * @param T             Sequence length
 * @param weights       SSM layer weights (shared across all K balls)
 * @param nested_state  Nested SSM state (contains K balls + curvatures)
 * @param conv_state    Conv1D state [B, CONV_KERNEL-1, CONV_DIM]
 * @param gating        Per-ball gating weights (optional, NULL = uniform)
 * @param output        Output tensor [B, T, D_MODEL]
 */
void wubu_nested_ssm_forward(const float *x, int B, int T,
                              const ssm_layer_weights *weights,
                              wubu_nested_ssm_state_t *nested_state,
                              float *conv_state,
                              const wubu_nested_ssm_gating_t *gating,
                              float *output);

/**
 * Initialize a nested SSM state with K balls and given curvatures.
 * States are initialized to zero (origin of each Poincaré ball).
 *
 * @param state  State struct to initialize
 * @param K      Number of balls
 * @param R      Array of K curvature radii
 * @return       0 on success, -1 on allocation failure
 */
int wubu_nested_ssm_init(wubu_nested_ssm_state_t *state, int K, const float *R);

/**
 * Free memory allocated by wubu_nested_ssm_init.
 */
void wubu_nested_ssm_free(wubu_nested_ssm_state_t *state);

/**
 * Check that all K balls' states are valid (no NaN, within radius).
 * Returns 0 if valid, 1 if any invalid state found.
 */
int wubu_nested_ssm_validate(const wubu_nested_ssm_state_t *state);

#ifdef __cplusplus
}
#endif

#endif // WUBU_NESTED_SSM_H
