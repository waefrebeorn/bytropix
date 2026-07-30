/* Test: EAGLE-style self-draft (doc 012).
 *
 * EAGLE-2/3 uses the target model's own early-layer features to draft
 * K tokens, then verifies them in a single forward pass via the
 * standard speculative acceptance mask (Leviathan et al.).
 * This test exercises the tree-draft + verify path end-to-end.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "wubu_spec_decode.h"

int main(void) {
    int vocab = 64;
    int K = 4;  /* draft depth */

    /* Build a target logit distribution where token 5 is the greedy pick
     * and tokens 10-13 are high-probability draft candidates. */
    float target_logits[64];
    for (int i = 0; i < vocab; i++) target_logits[i] = -100.0f;
    target_logits[5]  = 2.0f;   /* greedy accept */
    target_logits[10] = 1.5f;   /* draft candidate, high accept prob */
    target_logits[11] = 1.2f;
    target_logits[12] = 0.8f;
    target_logits[13] = 0.3f;   /* low, may reject */

    /* Draft probs: model is confident in 10, 11, 12; uncertain on 13. */
    float draft_probs[4] = {0.9f, 0.7f, 0.5f, 0.1f};
    int draft_tokens[4]  = {10, 11, 12, 13};
    int parent[4]        = {-1, 0, 1, 2};  /* tree: each draft depends on prior */

    /* Run one tree-draft verify step (Leviathan rejection sampling). */
    int accepted[16];
    int n_acc = wubu_spec_verify_tree(draft_tokens, parent, draft_probs,
                                               target_logits, K, vocab,
                                               accepted, 16, 0.5f);

    /* Tokens 10, 11, 12 should be accepted (draft prob close to target).
     * Token 13 likely rejected (low draft prob relative to its target). */
    printf("EAGLE self-draft: accepted %d/%d tokens\n", n_acc, K);
    for (int i = 0; i < n_acc; i++) {
        printf("  accepted[%d] = %d (draft_prob=%.2f target=%.2f)\n",
               i, accepted[i], draft_probs[i], target_logits[accepted[i]]);
    }

    /* Invariant: accepted tokens are a prefix of the draft tree. */
    for (int i = 0; i < n_acc; i++) {
        assert(accepted[i] == draft_tokens[i]);
    }

    printf("ALL EAGLE-SELF-DRAFT TESTS PASSED (n_acc=%d)\n", n_acc);
    return 0;
}
