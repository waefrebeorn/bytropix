/*
 * wubu_backprop.h -- the REAL backward pass + REAL Muon for the
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
 *     (the standard training-time memory cost). It runs the EXACT
 *     released path (wubu mode 0): partial RoPE per head,
 *     qk-norm, GQA, gated attention output, bounded SwiGLU, residual
 *     selectors, tied head.
 *   - BP2: the analytic backward pass, layer by layer, REVERSED:
 *     attention path (rope -> qk-norm -> softmax -> weighted sum ->
 *     o_proj/g_proj -> the gated residual), FFN path (swiglu ->
 *     gate_up -> down -> ffn_norm), the residual selectors, and the
 *     final norm + tied head. Every projection gets its REAL gradient.
 *   - BP3: the real Muon update: Nesterov momentum (0.95), then the
 *     Newton-Schulz 5 orthogonalization (a=3.4445, b=-4.7750,
 *     c=2.0315, Frobenius-normalized, tall matrices transposed), then
 *     the scaled step (the Moonlight per-matrix scale 0.2*sqrt(max
 *     dims)). Matrices = Muon; embeddings/norms/selectors (1-D) =
 *     AdamW -- the confirmed reference split.
 *   - BP4: a finite-difference verifier (tools/test_backprop.c) -- the
 *     DA doctrine: tests != correct, so the analytic gradients are
 *     checked against numerical gradients on a tiny model.
 */
#ifndef WUBU_BACKPROP_H
#define WUBU_BACKPROP_H

#include "wubu.h"
#include "wubu_train.h"

/* BP-A: the recorded activations for one sequence. The trainer owns
 * one of these; the forward fills it, the backward consumes it. */
typedef struct wubu_bp_t {
    int seq;
    int layers;   /* WUBU_LAYERS */
    int cap_seq;  /* the allocated sequence capacity (>= any seq used) */
    /* per-layer residual-stream snapshots: x BEFORE the layer */
    float *x_in;      /* [L, seq, 448] (the layer output survives) */
    float *emb_in;    /* [seq, 448] the embedding output (layer 0's
                         input and the first selector's checkpoint) */
    /* attention path */
    float *attn_norm; /* [L, seq, 448] rmsnorm(x_in) w/ attn_norm w */
    float *q_pre;     /* [L, seq, 448] q_proj out (pre qk-norm + rope,
                         needed by the norm backward) */
    float *k_pre;     /* [L, seq, 64]  k_proj out (pre k-norm + rope) */
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
    float *ffn_up;    /* [L, seq, 2456] the swiglu output (gate*up) */
    float *ffn_out;   /* [L, seq, 448] down(ffn_up) */
    /* selectors: the blend output per selector layer + the running
     * checkpoint (evolves every 4 layers). x_in[l] is left as the
     * layer's REAL output (pre-blend) so the layer backward is exact;
     * the blend itself lives in sel_out[l] and is what the next layer
     * consumes (and the checkpoint for the following selector). */
    float *sel_out;   /* [L, seq, 448] the blend at selector layers (0
                         for non-selector layers) */
    float *ckpt;      /* [seq, 448] the running group checkpoint
                         (ends as the LAST selector's blend) */
    float *sel_w0;    /* [L] the blend weight w0 for each layer (0 if
                         the layer has no selector) */
    /* the final hidden (pre lm_head) + the loss */
    float *final_h;   /* [seq, 448] the final-norm output */
    float *logits;    /* [seq, 16384] the head logits, computed ONCE per
                         forward by one GEMM (the old head_ce recomputed
                         them up to 5x per step -- the DA catch) */
    /* the softmax probs per (layer, head, position) are recomputed in
     * the backward from the saved q/k (memory-light: no extra store) */
    /* backward scratch (allocated once, reused per layer) */
    float *s_dq;      /* [seq, 448] dL/dq */
    float *s_dk;      /* [seq, 64]  dL/dk */
    float *s_dv;      /* [seq, 64]  dL/dv */
    float *s_dao;     /* [seq, 448] dL/dattn_out */
    float *s_dfg;     /* [seq, 2*FF] dL/dgate_up out (gate + up) */
    float *s_dfu;     /* [seq, FF]  dL/dffn_up */
    float *s_dfn;     /* [seq, 448] dL/dffn_norm out */
    float *s_dan;     /* [seq, 448] dL/dattn_norm out */
    float *s_dffn_out;/* [seq, 448] dL/dffn_out */
    float *s_do;      /* [seq, 448] dL/do_proj out */
    float *s_dg;      /* [seq, 448] dL/dg_proj out */
    float *s_dx;      /* [seq, 448] the incoming layer gradient */
    float *s_dxentry; /* [seq, 448] the gradient wrt the layer input */
} wubu_bp_t;

/* BP1: allocate the recorder for a given max sequence length. */
int wubu_bp_alloc(wubu_bp_t *bp, int max_seq);

/* BP2: the recording forward. Runs the exact released path (using the
 * buffer's rope tables -- b must be a wubu_buf_t whose tables were
 * built) and saves every activation the backward needs. Returns the
 * mean-reduced next-token cross-entropy loss. */
float wubu_bp_forward(wubu_model_t *m, wubu_buf_t *b, wubu_bp_t *bp,
                       const uint16_t *tokens, int n_tokens);

/* BP2b: same, but with an optional per-position loss mask (SFT
 * user-turn masking — the Smol Training Playbook's chat-template rule:
 * loss only on assistant tokens). mask[i] != 0 => position i is trained;
 * mask == NULL => train all positions (pretraining behavior). The mask
 * indexes the TARGET position (tokens[i] is predicted from tokens[i-1]). */
float wubu_bp_forward_masked(wubu_model_t *m, wubu_buf_t *b, wubu_bp_t *bp,
                             const uint16_t *tokens, int n_tokens,
                             const uint8_t *mask);

/* BP3: the analytic backward. Accumulates the REAL gradients into
 * tr (wubu_train_t), exactly like wubu_train_microbatch does.
 * Returns the loss (for the trainer's telemetry). */
float wubu_bp_backward(wubu_model_t *m, wubu_buf_t *b, wubu_bp_t *bp,
                        wubu_train_t *tr, const uint16_t *tokens,
                        int n_tokens);

/* BP3b: backward with the SFT user-turn mask (see wubu_bp_forward_masked). */
float wubu_bp_backward_masked(wubu_model_t *m, wubu_buf_t *b, wubu_bp_t *bp,
                              wubu_train_t *tr, const uint16_t *tokens,
                              int n_tokens, const uint8_t *mask);

/* BP4: the real optimizer step: Muon (Newton-Schulz 5) for the 2D
 * hidden matrices, AdamW for the embeddings, the norms and the
 * selectors (the confirmed reference split). Decoupled weight decay
 * for the Muon group; global-norm grad clipping per cfg->grad_clip. */
int wubu_bp_muon_step(wubu_model_t *m, wubu_train_t *tr,
                       const wubu_train_cfg_t *cfg, uint32_t step);

/* BP5: free. */
void wubu_bp_free(wubu_bp_t *bp);

#endif
