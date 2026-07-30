/*
 * wubu_attn_gate.h -- Attention-sink-free gated attention (doc 011).
 *
 * Per-channel gating: gate = σ(W_gate · x), applied to attention output.
 * Suppresses attention sinks (the disproportionate mass on token 0) and
 * eliminates the need for a no-op sink token.
 *
 * Self-contained C11, no third-party deps.
 */

#ifndef WUBU_ATTN_GATE_H
#define WUBU_ATTN_GATE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compute gate[D] = sigmoid(W_gate[D, hidden_dim] @ x[hidden_dim]).
 * Caller allocates gate[D]. */
void wubu_gate_sigmoid(float *gate, const float *x, const float *W_gate,
                        int D, int hidden_dim);

/* Apply per-channel gate to attention output: y[d] *= gate[d]. */
void wubu_apply_gate(float *attn_out, const float *gate, int D);

/* Full forward: y[D] = attn_out[D] * sigmoid(W_gate[D, hidden_dim] @ x).
 * D <= 16384 supported in this single-pass variant. */
void wubu_attn_gate_forward(const float *attn_out, const float *x,
                              const float *W_gate,
                              float *y, int D, int hidden_dim);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_ATTN_GATE_H */
