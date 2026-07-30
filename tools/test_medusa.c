/* Test: MEDUSA guess heads (doc 012).
 *
 * MEDusa uses multiple independent "guess heads" at different
 * layers to propose candidate tokens.  This test exercises the
 * multi-head draft + merge + verify path.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "wubu_spec_decode.h"

#define N_HEADS 3
#define DRAFT_DEPTH 4
#define VOCAB 32

int main(void) {
    /* Each head independently proposes a token from its top-1 guess. */
    int draft_tokens[DRAFT_DEPTH]  = {5, 12, 20, 7};
    float draft_probs[DRAFT_DEPTH][N_HEADS];
    float target_logits[VOCAB];

    /* Seed target logits so that heads 0 and 1 agree on token 5,
     * while head 2 has a different opinion (token 12). */
    for (int i = 0; i < VOCAB; i++) target_logits[i] = -100.0f;
    target_logits[5]  = 1.8f;  /* head 0 & 1 top pick */
    target_logits[12] = 1.4f;  /* head 2 top pick */
    target_logits[20] = 0.9f;
    target_logits[7]  = 0.3f;

    /* Head 0: picks token 5 confidently. */
    draft_probs[0][0] = 0.8f; draft_probs[0][1] = 0.05f; draft_probs[0][2] = 0.05f;
    /* Head 1: agrees with head 0 on token 5. */
    draft_probs[1][0] = 0.05f; draft_probs[1][1] = 0.85f; draft_probs[1][2] = 0.05f;
    /* Head 2: votes for token 12. */
    draft_probs[2][0] = 0.05f; draft_probs[2][1] = 0.05f; draft_probs[2][2] = 0.9f;
    /* Head 3 (last): low-confidence mixed vote. */
    draft_probs[3][0] = 0.4f; draft_probs[3][1] = 0.35f; draft_probs[3][2] = 0.25f;

    /* Merge scores across heads (weighted average). */
    float merged_probs[DRAFT_DEPTH];
    for (int i = 0; i < DRAFT_DEPTH; i++) {
        float sum = 0.0f;
        for (int h = 0; h < N_HEADS; h++) sum += draft_probs[i][h];
        merged_probs[i] = sum / N_HEADS;
    }

    /* Tree verify with merged probs. */
    int parent[DRAFT_DEPTH];
    for (int i = 0; i < DRAFT_DEPTH; i++) parent[i] = (i == 0) ? -1 : (i - 1);

    int accepted[DRAFT_DEPTH];
    int n_acc = wubu_spec_verify_tree(draft_tokens, parent, merged_probs,
                                                     target_logits, DRAFT_DEPTH, VOCAB,
                                                     accepted, DRAFT_DEPTH, 0.5f);

    printf("MEDUSA guess-heads: %d/%d accepted\n", n_acc, DRAFT_DEPTH);
    for (int i = 0; i < n_acc; i++) {
        printf("  accepted[%d] = %d (merged_prob=%.2f target=%.2f)\n",
               i, accepted[i], merged_probs[i], target_logits[accepted[i]]);
    }

    /* At least the first head (token 5) should be accepted. */
    assert(n_acc >= 1);
    assert(accepted[0] == 5);

    printf("ALL MEDUSA-GUESS-HEAD TESTS PASSED (n_acc=%d)\n", n_acc);
    return 0;
}
