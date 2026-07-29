/*
 * wubu_spec_cascade.h — Cascade speculative decoding (Area K).
 * Pure C11, self-contained, no third-party deps.
 */
#ifndef WUBU_SPEC_CASCADE_H
#define WUBU_SPEC_CASCADE_H

#include <stdint.h>
#include "wubu_ngram.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------- N-gram cascade (zero extra model) ---------- */

typedef struct wubu_ngram_cascade {
    wubu_ngram_draft_t *ngram;
    int draft_depth;
} wubu_ngram_cascade_t;

/* Create n-gram cascade drafter.
 * ctx/ctx_len: prompt context to seed n-gram statistics
 * order: n-gram order (2-5 typical)
 * draft_depth: tokens to draft per step (4-8)
 * defer_threshold: reserved for future use */
wubu_ngram_cascade_t *wubu_ngram_cascade_create(const int *ctx, int ctx_len, int order,
                                                 int draft_depth, int defer_threshold);
void wubu_ngram_cascade_free(wubu_ngram_cascade_t *c);

/* Propose up to draft_depth tokens. out_tokens filled, out_probs optional (uniform). */
int wubu_ngram_cascade_propose(wubu_ngram_cascade_t *c, int *out_tokens, float *out_probs);

/* Update n-gram context with accepted tokens. */
void wubu_ngram_cascade_update(wubu_ngram_cascade_t *c, const int *accepted, int n_accepted);

/* ---------- Self-cascade (small local model as drafter) ---------- */

typedef struct wubu_self_cascade {
    void *model_ctx;
    int (*forward)(void *model_ctx, const int *tokens, int n_tokens, float *logits, int vocab);
    int vocab;
    int draft_depth;
} wubu_self_cascade_t;

/* Small model forward signature: returns 0 on success, fills logits[vocab]. */
typedef int (*wubu_small_model_forward_fn)(void *model_ctx,
                                           const int *tokens, int n_tokens,
                                           float *logits, int vocab);

wubu_self_cascade_t *wubu_self_cascade_create(void *model_ctx,
                                               wubu_small_model_forward_fn forward,
                                               int vocab, int draft_depth,
                                               int defer_threshold);
void wubu_self_cascade_free(wubu_self_cascade_t *c);

/* Propose tokens using small model. out_probs optional (softmax of picked). */
int wubu_self_cascade_propose(wubu_self_cascade_t *c,
                               const int *context, int ctx_len,
                               int *out_tokens, float *out_probs);

/* ---------- Cascade verification with deferral ---------- */

/* Verify candidates against target logits with cascade deferral.
 * defer_eps: if draft_prob > target_prob * (1+eps), accept immediately (cascade speedup).
 * Returns number of accepted tokens. */
int wubu_cascade_verify(const int *candidates, const int *parent,
                        const float *draft_probs, const float *target_logits,
                        int n_cand, int vocab, int *accepted, int max_accepted,
                        float rng, float defer_eps);

/* ---------- High-level cascade step ---------- */

/* One decode iteration using n-gram cascade.
 * context: full context (prompt + generated so far)
 * target_logits: logits from target model at current position
 * Returns number of accepted tokens (0 = fallback to target). */
int wubu_cascade_step_ngram(wubu_ngram_cascade_t *drafter,
                            const int *context, int ctx_len,
                            const float *target_logits,
                            int *out_accepted, int max_accepted,
                            float rng, float defer_eps);

/* One decode iteration using self-cascade (small model drafter). */
int wubu_cascade_step_self(wubu_self_cascade_t *drafter,
                           const int *context, int ctx_len,
                           const float *target_logits,
                           int *out_accepted, int max_accepted,
                           float rng, float defer_eps);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SPEC_CASCADE_H */