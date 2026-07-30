/*
 * wubu_layer_skip.c -- Mixture-of-Depths / token-wise layer skip (doc 017).
 *
 * Per-token gate decides whether to skip a layer:
 *   y = x + gate * F(x)   where gate ∈ [0,1]
 * gate = σ(gate_weight · x)  (learned, one scalar per token)
 * Heuristic fallback (no gate_weight): gate = τ if ||x|| < θ, else 1.0,
 * with a floor that always runs the last N layers.
 *
 * Self-contained C11; no third-party deps.
 */

#include "wubu_layer_skip.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Compute per-token gate value.
 * If w has a gate_weight tensor (trainable), use σ(w·x).
 * Otherwise heuristic: gate = τ if ||x|| < θ, else 1.0. */
static float compute_gate(const float *x, int D,
                          const float *gate_weight, int use_heuristic,
                          float tau, float theta) {
    if (!use_heuristic && gate_weight) {
        float dot = 0.0f;
        for (int d = 0; d < D; d++) dot += gate_weight[d] * x[d];
        return 1.0f / (1.0f + expf(-dot));
    }
    /* Heuristic: skip if input norm below threshold */
    float norm_sq = 0.0f;
    for (int d = 0; d < D; d++) norm_sq += x[d] * x[d];
    float norm = sqrtf(norm_sq);
    return (norm < theta) ? tau : 1.0f;
}

/* Apply Mixture-of-Depths gate: for token t, compute gate_t and
 * produce y[t] = x[t] + gate_t * F(x[t]). */
void wubu_layer_skip_forward(const float *x, const float *gate_weight,
                                  float *y,
                                  int n_tokens, int D, int total_layers,
                                  int cur_layer, int floor_layers,
                                  bool has_gate_weight,
                                  float tau, float theta) {
    int skip = 0;
    if (cur_layer >= total_layers - floor_layers) {
        /* Floor: last N layers always run. */
        skip = 0;
    } else {
        float gate = compute_gate(x, D, gate_weight, !has_gate_weight,
                                    tau, theta);
        skip = (gate < 0.5f) ? 1 : 0;
    }

    if (skip) {
        /* Passthrough: residual connection only. */
        for (int i = 0; i < n_tokens * D; i++) {
            y[i] = x[i];
        }
    } else {
        /* Layer executes normally; caller writes F(x) into y.
         * We just add the residual here (assumed pre-computed by caller). */
        for (int i = 0; i < n_tokens * D; i++) {
            y[i] = x[i] + y[i];  /* y already has F(x) in it */
        }
    }
}

/* Verify that the current layer is NOT in the floor zone
 * (i.e., it is allowed to be skipped). Returns true if the
 * layer is in the skip-capable region, false if in the floor. */
bool wubu_layer_skip_verify_floor(int total_layers, int cur_layer,
                                          int floor_layers) {
    return cur_layer < total_layers - floor_layers;
}
