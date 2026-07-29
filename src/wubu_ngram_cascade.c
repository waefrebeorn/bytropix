/*
 * wubu_ngram_cascade.c — N-gram cascade drafter (zero extra model).
 * Pure C11, self-contained. Uses prompt n-gram statistics to draft tokens.
 */
#include "wubu_spec_cascade.h"
#include "wubu_ngram.h"
#include <stdlib.h>
#include <string.h>

/* Use the wubu_ngram_cascade_t struct defined in wubu_spec_cascade.h */

wubu_ngram_cascade_t *wubu_ngram_cascade_create(const int *ctx, int ctx_len, int order,
                                                 int draft_depth, int defer_threshold) {
    (void)defer_threshold;
    wubu_ngram_cascade_t *c = (wubu_ngram_cascade_t *)malloc(sizeof(*c));
    if (!c) return NULL;
    c->ngram = wubu_ngram_create(ctx, ctx_len, order);
    c->draft_depth = draft_depth > 0 ? draft_depth : 4;
    return c;
}

void wubu_ngram_cascade_free(wubu_ngram_cascade_t *c) {
    if (c) {
        if (c->ngram) wubu_ngram_free(c->ngram);
        free(c);
    }
}

int wubu_ngram_cascade_propose(wubu_ngram_cascade_t *c, int *out_tokens, float *out_probs) {
    if (!c || !c->ngram) return 0;
    int proposed = wubu_ngram_propose(c->ngram, c->draft_depth, out_tokens);
    if (proposed > 0 && out_probs) {
        for (int i = 0; i < proposed; i++) out_probs[i] = 1.0f / proposed;
    }
    return proposed;
}

void wubu_ngram_cascade_update(wubu_ngram_cascade_t *c, const int *accepted, int n_accepted) {
    if (!c || !c->ngram) return;
    wubu_ngram_update_context(c->ngram, accepted, n_accepted);
}