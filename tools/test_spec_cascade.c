/* Test: CAS-Spec adaptive deferral (doc 018).
 *
 * Validates the defer_threshold / adaptive-deferral path in
 * wubu_ngram_cascade_create by verifying that:
 *  1. A cascade with defer_threshold=0 drafts eagerly (no deferral).
 *  2. A cascade with a non-zero defer_threshold still produces at least
 *     one draft token from a repeating context.
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "wubu_spec_cascade.h"

int main(void) {
    int ctx[16] = {1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1};
    int ctx_len = 16;

    /* Eager mode (defer_threshold=0). */
    wubu_ngram_cascade_t *eager = wubu_ngram_cascade_create(ctx, ctx_len, 3, 4, 0.0f);
    assert(eager);
    int tokens_eager[16];
    int n_eager = wubu_ngram_cascade_propose(eager, tokens_eager, NULL);
    assert(n_eager > 0);
    printf("EAGER   (defer=0): drafted %d tokens\n", n_eager);
    wubu_ngram_cascade_free(eager);

    /* Adaptive mode (defer_threshold>0). */
    wubu_ngram_cascade_t *adapt = wubu_ngram_cascade_create(ctx, ctx_len, 3, 4, 0.5f);
    assert(adapt);
    int tokens_adapt[16];
    int n_adapt = wubu_ngram_cascade_propose(adapt, tokens_adapt, NULL);
    assert(n_adapt > 0);
    printf("ADAPTIVE (defer=0.5): drafted %d tokens\n", n_adapt);
    wubu_ngram_cascade_free(adapt);

    printf("ALL CAS-SPEC ADAPTIVE DEFERRAL TESTS PASSED (eager=%d adapt=%d)\n", n_eager, n_adapt);
    return 0;
}
