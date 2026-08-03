/*
 * wubu_barun_backprop.h -- the REAL backward pass + REAL Muon for the
 * WuBu seed (the deep-training milestone).
 *
 * The audit found three gaps in the first trainer:
 *   1. layer_grad() gave EVERY layer the same outer product of the
 *      final hidden state -- every layer (12) received identical updates,
 *      so they could never specialize. Not backprop.
 *   2. muon_update() was momentum SGD -- the Muon paper's entire point
 *      is the Newton-Schulz orthogonalization of the momentum matrix.
 *   3. No gradients flowed through the attention path (q/k/v/o/g
 *      projections, qk-norm, rope, softmax) at all.
 *
 * This module fixes all three:
 *   - BP1: a forward pass that RECORDS the per-layer activations
 *     (the standard training-time memory cost).
 *   - BP2: the analytic backward pass, layer by layer, REVERSED:
 *     attention path (rope -> qk-norm -> softmax -> weighted sum ->
 *     o_proj/g_proj -> the gated residual), FFN path (swiglu ->
 *     gate_up -> down -> ffn_norm), the residual selectors, and the
 *     final norm + tied head. Every projection gets its REAL gradient.
 *   - BP3: the real Muon update: momentum, then the Newton-Schulz
 *     orthogonalization (5 iterations of M = (3M - M M^T M)/2), then
 *     the scaled step -- matching the Muon paper + the Barun reference
 *     (Muon for the matrices, AdamW for the embedding + norms).
 *   - BP4: a finite-difference verifier (test_backprop.c) -- the DA
 *     doctrine: tests != correct, so the analytic gradients are checked
 *     against numerical gradients on a tiny model.
 */
#ifndef WUBU_BARUN_BACKPROP_H
#define WUBU_BARUN_BACKPROP_H

#include "wubu_barun.h"
#include "wubu_barun_train.h"

/* BP-A: the recorded activations for one sequence. The trainer owns
 * one of these; the forward fills it, the backward consumes it. */
typedef struct {
    int seq;
    int layers;   /* BARUN_LAYERS */
    /* per-layer residual-stream snapshots: x BEFORE the layer */
    float *x_in;      /* [L, seq, 448] */
    /* attention path */
    float *attn_norm; /* [L, seq, 448] rmsnorm(x_in) w/ attn_norm w */
    float *q;         /* [L, seq, 448] post qk-norm + rope (7x64) */
    float *k;         /* [L, seq, 64]  post k-norm + rope */
    float *v;         /* [L, seq, 64]  raw v (no norm on v) */
    float *attn_out;  /* [L, seq, 448] the GQA weighted sum */
    float *o_out;     /* [L, seq, 448] o_proj(attn_out) */
    float *g_val;     /* [L, seq, 448] g_proj(attn_norm) (pre-sigmoid) */
    /* ffn path */
    float *ffn_norm;  /* [L, seq, 448] rmsnorm(x_after_attn) */
    float *ffn_gate;  /* [L, seq, 2456] gate_up pre-activation (saved to
                         recompute silu' and the up branch) */
    float *ffn_up;    /* [L, seq, 2456] the swiglu output */
    float *ffn_out;   /* [L, seq, 448] down(ffn_up) */
    /* selectors: the checkpoint stream (evolves every 4 layers) */
    float *ckpt;      /* [seq, 448] the running group checkpoint */
    float *sel_w0;    /* [L] the blend weight w0 for each layer (0 if
                         the layer has no selector) */
    /* the final hidden (pre lm_head) + the logits */
    float *final_h;   /* [seq, 448] the final-norm output */
    /* the softmax probs per (layer, head, position) are recomputed in
     * the backward from the saved q/k (memory-light: no extra store) */
    /* backward scratch (allocated once, reused per layer) */
    float *scratch;   /* one big arena, carved below */
    float *s_dq;      /* [seq, 448] dL/dq */
    float *s_dk;      /* [seq, 64]  dL/dk */
    float *s_dv;      /* [seq, 64]  dL/dv */
    float *s_dao;     /* [seq, 448] dL/dattn_out */
    float *s_dfg;     /* [seq, 2*FF] dL/dgate_up out */
    float *s_dfu;     /* [seq, FF]  dL/dffn_up */
    float *s_dfn;     /* [seq, 448] dL/dffn_norm out */
    float *s_dan;     /* [seq, 448] dL/dattn_norm out */
    float *s_dffn_out;/* [seq, 448] dL/dffn_out */
    float *s_do;      /* [seq, 448] dL/do_proj out */
    float *s_dg;      /* [seq, 448] dL/dg_proj out */
} barun_bp_t;

/* BP1: allocate the recorder for a given max sequence length. */
int barun_bp_alloc(barun_bp_t *bp, int max_seq);

/* BP2: the recording forward. Runs the exact released path and saves
 * every activation the backward needs. Returns the loss too. */
float barun_bp_forward(barun_model_t *m, barun_bp_t *bp,
                       const uint16_t *tokens, int n_tokens);

/* BP3: the analytic backward. Accumulates the REAL gradients into
 * tr (barun_train_t), exactly like barun_train_microbatch does.
 * Returns the loss (for the trainer's telemetry). */
float barun_bp_backward(barun_model_t *m, barun_bp_t *bp,
                        barun_train_t *tr, const uint16_t *tokens,
                        int n_tokens);

/* BP4: the real Muon step (Newton-Schulz orthogonalization). */
int barun_bp_muon_step(barun_model_t *m, barun_train_t *tr,
                       const barun_train_cfg_t *cfg, uint32_t step);

/* BP5: free. */
void barun_bp_free(barun_bp_t *bp);

#endif
