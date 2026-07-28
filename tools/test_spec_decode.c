/* Test: wubu_spec_decode (Area A — speculative decoding). */
#include "wubu_spec_decode.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

static int approx(float a, float b) { return fabsf(a - b) < 1e-4f; }

int main(void) {
    int vocab = 10;
    /* Synthetic target distribution: token 3 is most likely. */
    float target[10];
    for (int i = 0; i < vocab; i++) target[i] = 0.01f;
    target[3] = 0.80f; target[5] = 0.10f;

    /* --- Tree verify: draft proposes [3, 5] in sequence. --- */
    int cand[4] = {3, 5, 3, 5};
    int par[4]  = {-1, 0, -1, 2};           /* 0 root, 1 child of 0, 2 root, 3 child of 2 */
    float dp[4] = {0.7f, 0.6f, 0.7f, 0.6f}; /* draft confident */
    int acc[8];

    int a1 = wubu_spec_verify_tree(cand, par, dp, target, 4, vocab, acc, 8, 0.0f);
    /* pos0: cand0=3, p_target=0.8 >= p_draft=0.7 -> accept 3.
       pos1: cand1=5 child of 0 (accepted), p_target=0.10 < p_draft=0.6, rng=0 < 1/6 -> accept 5.
       pos2: cand2=3 root (accepted_n=2, parent -1 accepted) accept 3.
       pos3: cand3=5 child of 2 (accepted) accept 5. -> all 4 accepted. */
    printf("tree-verify accepted=%d (expect 4)\n", a1);
    assert(a1 == 4);
    assert(acc[0] == 3 && acc[1] == 5 && acc[2] == 3 && acc[3] == 5);

    /* --- Rejection case: draft proposes token 7 (target ~0). --- */
    int cand2[1] = {7}; int par2[1] = {-1}; float dp2[1] = {0.9f};
    float rng = 0.5f;
    /* p_target(7)=0.01, p_draft=0.9, p_target/p_draft=0.011; rng=0.5 > 0.011 -> reject */
    int a2 = wubu_spec_verify_tree(cand2, par2, dp2, target, 1, vocab, acc, 8, rng);
    printf("reject-case accepted=%d (expect 0)\n", a2);
    assert(a2 == 0);

    /* --- n-gram draft --- */
    int ctx[8] = {1, 2, 3, 1, 2, 3, 1, 2};   /* repeating 1,2,3 */
    wubu_ngram_draft_t *ng = wubu_ngram_create(ctx, 8, 3);
    int out[4];
    int nprop = wubu_ngram_propose(ng, 3, out);
    printf("ngram proposed=%d first=%d (expect 1, tok=3)\n", nprop, out[0]);
    assert(nprop >= 1 && out[0] == 3);
    wubu_ngram_free(ng);

    /* --- bonus token: residual of target - draft --- */
    float draft[10]; for (int i = 0; i < vocab; i++) draft[i] = 0.0f;
    draft[3] = 0.7f;  /* draft also likes 3 */
    int bonus = wubu_spec_bonus_token(target, draft, vocab, 0.999f);
    printf("bonus token=%d\n", bonus);
    assert(bonus >= 0 && bonus < vocab);

    printf("ALL SPEC-DECODE TESTS PASSED\n");
    return 0;
}
