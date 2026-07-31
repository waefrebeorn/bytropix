/*
 * wubu_attn_gate.c -- Attention-sink-free gated attention (doc 011).
 *
 * Conditional gating: gate = σ(W_gate · x) applied per-channel/per-head
 * *dynamically* to the attention output. Suppresses attention sinks and
 * eliminates the "no-op" sink token (paper: arXiv:2603.05498 + NeurIPS'25).
 *
 * Self-contained C11. Tested via tools/test_attn_gate.c.
 */

#include "wubu_attn_gate.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Per-channel gate value, computed once and broadcast. */
void wubu_gate_sigmoid(float *gate, const float *x, const float *W_gate,
                        int D, int hidden_dim) {
    /* gate[D] = sigmoid(W_gate[D, hidden_dim] @ x[hidden_dim]) */
    for (int d = 0; d < D; d++) {
        float dot = 0.0f;
        const float *wr = W_gate + (size_t)d * hidden_dim;
        for (int k = 0; k < hidden_dim; k++) {
            dot += wr[k] * x[k];
        }
        gate[d] = 1.0f / (1.0f + expf(-dot));
    }
}

/* Apply per-channel gate to attention output: y[d] = attn_out[d] * gate[d]. */
void wubu_apply_gate(float *attn_out, const float *gate, int D) {
    for (int d = 0; d < D; d++) {
        attn_out[d] *= gate[d];
    }
}

/* Full forward: softmax·V → σ(W_gate·x) → elementwise mul.
 * Uses heap allocation for D > 4096 (no stack overflow). */
void wubu_attn_gate_forward(const float *attn_out, const float *x,
                              const float *W_gate,
                              float *y, int D, int hidden_dim) {
    float gate_stack[4096];
    float *gate = gate_stack;
    if (D > 4096) {
        gate = (float *)malloc(sizeof(float) * D);
        if (!gate) {
            /* OOM: copy attn_out to y ungated (degraded but not crashed) */
            memcpy(y, attn_out, sizeof(float) * D);
            return;
        }
    }
    wubu_gate_sigmoid(gate, x, W_gate, D, hidden_dim);
    memcpy(y, attn_out, sizeof(float) * D);
    wubu_apply_gate(y, gate, D);
    if (gate != gate_stack) free(gate);
}
