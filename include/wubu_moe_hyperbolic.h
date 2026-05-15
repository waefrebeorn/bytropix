#ifndef WUBU_MOE_HYPERBOLIC_H
#define WUBU_MOE_HYPERBOLIC_H

#include <stdbool.h>
#include <stdint.h>

#include "wubu_moe.h"   // N_EXPERTS, N_ACTIVE_EXPTS, D_FF
#include "wubu_mobius.h" // wubu_poincare_dist
#include "wubu_ssm.h"    // D_MODEL

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Nested MoE: Poincaré distance router + 2-level hierarchy
// ============================================================

// Hierarchy parameters
#define N_HYPERBOLIC_GROUPS     16    // number of coarse groups
#define N_EXPERTS_PER_GROUP     16    // experts per group (N_EXPERTS = 256 = 16*16)
#define R_POINCARE_COARSE       1.5f  // Poincaré ball radius for coarse level
#define R_POINCARE_FINE         0.5f  // Poincaré ball radius for fine level
#define R_POINCARE_INPUT        1.5f  // Poincaré ball radius for input mapping (same as coarse)
#define HYPERBOLIC_TEMPERATURE  0.1f  // temperature for distance-to-score conversion

// ============================================================
// Poincaré distance router
// ============================================================

// Centroid storage for flat (single-level) Poincaré router
// centroids: [N_EXPERTS, D_MODEL] — each row is a centroid in the Poincaré ball
typedef struct {
    float *centroids;     // [N_EXPERTS * D_MODEL] — Poincaré ball centroids
    float temperature;    // scaling for distance → score conversion
    bool loaded;
} poincare_router_t;

// Initialize Poincaré router with synthetic random centroids
// centroids_out: caller-provided buffer [N_EXPERTS * D_MODEL]
void wubu_poincare_router_init_random(float *centroids_out, unsigned int seed);

// Poincaré distance router: replace linear x @ gate_inp with hyperbolic routing
// x: [B, T, D_MODEL] — input (Euclidean, will be mapped to Poincaré ball)
// router: centroids + temperature
// scores: [B*T, N_EXPERTS] — output scores (pre-softmax, negative distances / temp)
void wubu_poincare_router_forward(const float *x, int B, int T,
                                  const poincare_router_t *router,
                                  float *scores);

// ============================================================
// 2-level hierarchical routing
// ============================================================

// Two-level hierarchy centroids
// coarse_centroids: [N_HYPERBOLIC_GROUPS, D_MODEL] in Poincaré ball (R=1.5)
// fine_centroids: [N_EXPERTS, D_MODEL] in Poincaré ball (R=0.5)
//   fine_centroids[group * N_EXPERTS_PER_GROUP * D_MODEL + ...]
typedef struct {
    float *coarse_centroids;  // [N_HYPERBOLIC_GROUPS * D_MODEL]
    float *fine_centroids;    // [N_EXPERTS * D_MODEL]
    float temperature;
    bool loaded;
} nested_moe_router_t;

// Initialize nested MoE router with synthetic random centroids
void wubu_nested_moe_router_init_random(float *coarse_out, float *fine_out,
                                        unsigned int seed);

// Two-level hierarchical routing
// Level 1: Poincaré distance to 16 coarse centroids (R=1.5) → pick top-1 group
// Level 2: Poincaré distance to 16 fine centroids in selected group (R=0.5) → pick top-2
// Combined: 32 candidates → score re-ranking → pick top-8
//
// x: [B, T, D_MODEL] — input
// router: coarse + fine centroids
// out_indices: [B*T, N_ACTIVE_EXPTS] — selected expert indices
// out_weights: [B*T, N_ACTIVE_EXPTS] — normalized weights
void wubu_nested_moe_router_forward(const float *x, int B, int T,
                                    const nested_moe_router_t *router,
                                    int *out_indices, float *out_weights);

// Free router resources
void wubu_poincare_router_free(poincare_router_t *router);
void wubu_nested_moe_router_free(nested_moe_router_t *router);

// ============================================================
// Poincaré router backward pass (single-level)
// ============================================================
// Backprop through Poincaré distance routing.
// Straight-through: top-k selection is treated as non-differentiable.
// Only gradient through the assigned scores flows back to centroids and input.
// x:        [B*T, D_MODEL] — forward input (Euclidean, same as forward)
// scores:   [B*T, N_EXPERTS] — forward output (pre-softmax scores)
// d_scores: [B*T, N_EXPERTS] — upstream gradient
// router:   centroids + temperature
// d_x:      [B*T, D_MODEL] — gradient w.r.t. input (add to existing, or NULL)
// d_centroids: [N_EXPERTS * D_MODEL] — gradient w.r.t. centroids (or NULL)
void wubu_poincare_router_backward(const float *x, int B, int T,
                                   const float *scores,
                                   const float *d_scores,
                                   const poincare_router_t *router,
                                   float *d_x,
                                   float *d_centroids);

// ============================================================
// Two-level nested MoE router backward (straight-through estimation)
// ============================================================
// x:        [B*T, D_MODEL] — forward input (Euclidean)
// out_indices:  [B*T, N_ACTIVE_EXPTS] — forward selected expert indices
// out_weights:  [B*T, N_ACTIVE_EXPTS] — forward final normalized weights
// d_out_weights: [B*T, N_ACTIVE_EXPTS] — upstream gradient w.r.t. final weights
// router:   coarse + fine centroids + temperature
// d_x:      [B*T, D_MODEL] — gradient w.r.t. input (accumulate, or NULL)
// d_coarse_centroids: [N_HYPERBOLIC_GROUPS * D_MODEL] — gradient (or NULL)
// d_fine_centroids: [N_EXPERTS * D_MODEL] — gradient (or NULL)
void wubu_nested_moe_router_backward(
    const float *x, int B, int T,
    const int *out_indices, const float *out_weights,
    const float *d_out_weights,
    const nested_moe_router_t *router,
    float *d_x,
    float *d_coarse_centroids,
    float *d_fine_centroids);

// ============================================================
// Utility functions (also usable from test code)
// ============================================================

// Map Euclidean vector to Poincaré ball via exp_map
// v: input vector (any magnitude)
// d: dimension
// R: ball radius
// out: output in Poincaré ball (||out|| < R)
void euclidean_to_poincare_ball(const float *v, int d, float R, float *out);

// Top-k selection from an array (indices + values)
void topk_from_array(const float *vals, int n, int k,
                     int *out_indices, float *out_vals);

#ifdef __cplusplus
}
#endif

#endif // WUBU_MOE_HYPERBOLIC_H
