/* Test: J03 early-exit + self-speculative verify.
 * Verifies: (1) early-exit triggers on low hidden delta, disabled when threshold
 * is huge; (2) draft only continues while shallow==full; (3) verify accepts the
 * leading run of high-prob tokens and stops at first rejection. */
#include "wubu_early_exit.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

int main(void) {
    printf("=== J03 Early-Exit + Self-Spec Test ===\n");

    /* Disabled gate */
    wubu_early_exit_t *off = wubu_early_exit_create(1e30f, 0);
    assert(!wubu_early_exit_should_stop(off, 50, 80, 0.0f, 5.0f));
    wubu_early_exit_free(off);

    /* Enabled gate: tiny relative delta -> exit */
    wubu_early_exit_t *on = wubu_early_exit_create(0.01f, 4);
    int stop = wubu_early_exit_should_stop(on, 60, 80, 0.02f, 5.0f); /* rel=0.004 */
    printf("  early-exit at layer 60: %s\n", stop ? "YES" : "no");
    assert(stop == 1);
    /* First 25% of layers never exits */
    assert(wubu_early_exit_should_stop(on, 5, 80, 0.0f, 5.0f) == 0);

    /* Self-spec draft: shallow==full for first 3, then diverges */
    int shallow[4] = {7, 7, 7, 9};
    int full[4]    = {7, 7, 7, 3};
    int draft[4];
    int nd = wubu_early_exit_draft(4, shallow, full, draft);
    printf("  drafted %d tokens (expect 3)\n", nd);
    assert(nd == 3);

    /* Verify: probs high for first 2, then drop below threshold */
    float probs[3] = {0.9f, 0.8f, 0.1f};
    int accepted = 0;
    int all = wubu_early_exit_verify(on, draft, probs, 3, 0.5f, &accepted);
    printf("  verify accepted=%d all_accepted=%s (expect 2, no)\n", accepted, all ? "yes" : "no");
    assert(accepted == 2 && all == 0);

    wubu_early_exit_free(on);
    printf("ALL J03 EARLY-EXIT TESTS PASSED\n");
    return 0;
}
