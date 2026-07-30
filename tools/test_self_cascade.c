/* Test: self-cascade drafter (doc 012/018). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "wubu_spec_cascade.h"

int main(void) {
    /* Repeating 3-gram to give the cascade drafter unambiguous matches. */
    int ctx[16] = {1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1};
    int ctx_len = 16;

    /* Phase A: n-gram cascade (doc 018 flavor 1). */
    wubu_ngram_cascade_t *cascade = wubu_ngram_cascade_create(ctx, ctx_len, 3, 4, 0.1f);
    assert(cascade != NULL);

    int tokens[16];
    int n = wubu_ngram_cascade_propose(cascade, tokens, NULL);
    assert(n > 0);
    printf("NGram cascade drafted %d tokens\n", n);

    wubu_ngram_cascade_update(cascade, tokens, n);
    wubu_ngram_cascade_free(cascade);

    printf("ALL SELF-CASCADE TESTS PASSED (n_draft=%d)\n", n);
    return 0;
}
