#ifndef WUBU_POINCARE_GQA_H
#define WUBU_POINCARE_GQA_H

/**
 * Poincaré GQA (hyperbolic distance attention) forward pass.
 *
 * Replaces standard softmax dot-product attention with hyperbolic
 * distance-based attention in the Poincaré ball:
 *
 * 1. Q/K/V projections (same as Euclidean GQA)
 * 2. Map Q, K, V to Poincaré ball via exp_map(Q, R)
 * 3. Attention score = softmax(-d(q_ball, k_ball) / tau)
 *    where d is the Poincaré geodesic distance
 * 4. Output in ball = Möbius combination of V_ball weighted by scores
 * 5. Map output back via log_map
 *
 * All other steps (Q+gate fused projection, K/V projection,
 * RMSNorm, gate sigmoid, output projection) are identical to
 * the Euclidean GQA forward pass.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "wubu_ssm.h"

// ============================================================
// Poincaré GQA Forward Pass
// ============================================================
void wubu_poincare_gqa_forward(const float *x, int B, int T,
                               const gqa_layer_weights *weights,
                               float R,
                               float *output);

#ifdef __cplusplus
}
#endif

#endif // WUBU_POINCARE_GQA_H
