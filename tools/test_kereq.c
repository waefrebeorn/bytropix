/* Test: wubu_kereq (Round-2 #121 — kernel equivalence prover). */
#include "wubu_kereq.h"
#include <stdio.h>
#include <assert.h>

int main(void) {
    float cx;
    /* Correct candidate: proven equal over [-3,3] with scale=0.5,bias=0,clamp[0,1]. */
    int eq_ok = wubu_kereq_prove_eq(-3.0f, 3.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0, &cx);
    printf("correct kernel proven equal = %d (expect 1)\n", eq_ok);
    assert(eq_ok == 1);

    /* Buggy candidate (upper clamp off by 1e-7): diverges near x=2 where
     * 0.5*2 = 1.0 hits the clamp boundary. Prover returns SAT (0) + counterexample. */
    int eq_bug = wubu_kereq_prove_eq(-3.0f, 3.0f, 0.5f, 0.0f, 0.0f, 1.0f, 1, &cx);
    printf("buggy kernel proven equal = %d (expect 0 / SAT), cx=%.8f\n", eq_bug, cx);
    assert(eq_bug == 0);
    assert(cx > 0.999f && cx <= 1.0f);  /* counterexample at the clamp boundary */

    printf("ALL KEREQ TESTS PASSED\n");
    return 0;
}
