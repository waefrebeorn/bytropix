/*
 * wubu_ngram.h — N-gram drafter for speculative decoding (cascade & fallback).
 * Pure C11, self-contained, zero external model weights.
 *
 * Builds a rolling n-gram index from the prompt context and proposes
 * the longest-matching continuation. 2-3× speedup on matched prompts.
 */
#ifndef WUBU_NGRAM_H
#define WUBU_NGRAM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_ngram_draft {
    int *ctx;
    int ctx_len;
    int cap;
    int order;
} wubu_ngram_draft_t;

/* Create n-gram drafter from context tokens.
 * order: n-gram order (2=bigram, 3=trigram, etc.)
 * Returns opaque drafter or NULL on OOM. */
wubu_ngram_draft_t *wubu_ngram_create(const int *ctx, int ctx_len, int order);

/* Free drafter. */
void wubu_ngram_free(wubu_ngram_draft_t *d);

/* Propose up to `k` draft tokens by extending the longest matching
 * n-gram suffix of the context. Returns number proposed; fills out[]. */
int wubu_ngram_propose(wubu_ngram_draft_t *d, int k, int *out);

/* Update the n-gram context with newly accepted tokens. */
void wubu_ngram_update_context(wubu_ngram_draft_t *d, const int *accepted, int n_accepted);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_NGRAM_H */