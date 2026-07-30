/*
 * wubu_layer_skip.h -- Mixture-of-Depths / token-wise layer skip (doc 017).
 *
 * Per-token gate decides whether to skip a layer's compute.
 * Learned gate (trainable) or heuristic (||x|| < θ ⇒ skip).
 * Floor: last N layers always run regardless of gate.
 *
 * Self-contained C11; no third-party deps.
 */

#ifndef WUBU_LAYER_SKIP_H
#define WUBU_LAYER_SKIP_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Apply Mixture-of-Depths layer skip on a per-token basis.
 *
 * x:          [n_tokens * D] input activations
 * gate_weight:[D]          learned gate weight (NULL → heuristic mode)
 * y:          [n_tokens * D] output (F(x) pre-added by caller)
 * n_tokens:   number of tokens in batch
 * D:          model dimension
 * total_layers: total number of layers in the model
 * cur_layer:  current layer index (0-based)
 * floor_layers: number of layers that always run (never skipped)
 * has_gate_weight: true if gate_weight is a learned (trainable) parameter
 * tau:        heuristic tau value for gate < 0.5 → skip
 * theta:      heuristic ||x|| threshold below which to skip
 */
void wubu_layer_skip_forward(const float *x, const float *gate_weight,
                                  float *y,
                                  int n_tokens, int D, int total_layers,
                                  int cur_layer, int floor_layers,
                                  bool has_gate_weight,
                                  float tau, float theta);

/* Verify that the floor constraint is respected: returns true if
 * cur_layer should be considered (not in the floor zone). */
bool wubu_layer_skip_verify_floor(int total_layers, int cur_layer,
                                          int floor_layers);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_LAYER_SKIP_H */
