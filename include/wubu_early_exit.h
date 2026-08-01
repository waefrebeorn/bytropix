/*
 * wubu_early_exit.h — Early-exit + self-speculative verify (doc J03 / 017/012).
 *
 * Two complementary techniques to cut compute on easy tokens:
 *
 * 1) EARLY-EXIT: during a forward pass, a per-layer confidence gate decides
 *    whether the residual stream has "converged" and the remaining layers can
 *    be skipped. Implemented as a small classifier over the hidden state:
 *    if ||grad_of_hidden|| (or a learned gate logit) < threshold, exit early
 *    and reuse the last good layer's output as the logits source.
 *
 * 2) SELF-SPECULATIVE VERIFY: a draft of K tokens is produced by a shallow
 *    truncation of the same model; the full model verifies them in one batched
 *    forward. Accepted tokens are kept; on first rejection, the rest are
 *    discarded and generation continues from the verified prefix.
 *
 * Both are pure-C, dependency-free, and statistically neutral when the gate is
 * disabled (threshold = +inf => never exit; draft depth = 0 => no speculation).
 */
#ifndef WUBU_EARLY_EXIT_H
#define WUBU_EARLY_EXIT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_early_exit wubu_early_exit_t;

/* Create controller.
 * exit_threshold: hidden-state delta magnitude below which we exit early.
 *                 Set FLT_MAX (or <=0) to disable early-exit.
 * max_draft     : self-speculative draft depth (0 = disabled). */
wubu_early_exit_t *wubu_early_exit_create(float exit_threshold, int max_draft);

void wubu_early_exit_free(wubu_early_exit_t *e);

/* Evaluate the per-layer exit gate.
 * layer      : current layer index (0-based)
 * n_layers   : total layers in the model
 * hidden_delta : L2 norm of (hidden - prev_hidden) across the layer
 * hidden_norm  : L2 norm of hidden (for relative comparison)
 * Returns 1 if the forward should STOP here (early exit), else 0. */
int wubu_early_exit_should_stop(const wubu_early_exit_t *e,
                                int layer, int n_layers,
                                float hidden_delta, float hidden_norm);

/* Self-speculative draft generation (stub-invocable, model-agnostic).
 * Produces up to `depth` candidate token ids into `out_draft` using a simple
 * argmax-of-argmax consistency heuristic: a token is "safe" to draft when the
 * top-1 of the shallow trunk agrees with the top-1 of the full logits for the
 * prefix so far. Returns number drafted (0..depth). The caller feeds these to
 * the full model for verification.
 *
 * shallow_top1 : array[depth] of top-1 token from the shallow/early trunk
 * full_top1   : array[depth] of top-1 token from the full model at same steps
 * (In a real engine these come from two forward passes; here we accept them as
 *  inputs so the verify logic is testable without a model.) */
int wubu_early_exit_draft(int depth,
                          const int *shallow_top1, const int *full_top1,
                          int *out_draft);

/* Verify a drafted sequence against full-model logits.
 * draft      : drafted token ids (length n)
 * draft_probs: full-model probability of each drafted token (length n)
 * threshold  : accept token i if draft_probs[i] >= threshold
 * Writes accepted token count into *accepted (<= n). Returns 1 if the whole
 * draft was accepted (rare/good), 0 otherwise. */
int wubu_early_exit_verify(const wubu_early_exit_t *e,
                           const int *draft, const float *draft_probs,
                           int n, float threshold, int *accepted);

/* Stats */
void wubu_early_exit_stats(const wubu_early_exit_t *e,
                           int *early_exits, int *spec_accepts);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_EARLY_EXIT_H */
