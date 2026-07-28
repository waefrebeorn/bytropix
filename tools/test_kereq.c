/* Test: wubu_kereq (Round-2 #121 — genuine symbolic equivalence prover).
 * Tests the real abstract-interpretation prover, plus DA edge cases. */
#include "wubu_kereq.h"
#include <stdio.h>
#include <assert.h>

int main(void) {
    float cx;
    /* Identical specs -> proven EQUAL (UNSAT). */
    wubu_affine_clamp_t ref = {0.5f, 0.0f, 0.0f, 1.0f};
    wubu_affine_clamp_t same = {0.5f, 0.0f, 0.0f, 1.0f};
    int r = wubu_kereq_prove_eq(&ref, &same, -3.0f, 3.0f, &cx);
    printf("identical kernels -> %d (expect 1 EQUAL)\n", r);
    assert(r == 1);

    /* Genuinely different (disjoint) clamp -> proven DIVERGENT (SAT).
     * cand clamps to [1.1, 2.0]; ref output [0,1.0] is disjoint -> at any
     * x the candidate's range never meets the reference's. */
    wubu_affine_clamp_t cand = {0.5f, 0.0f, 1.1f, 2.0f};
    r = wubu_kereq_prove_eq(&ref, &cand, -3.0f, 3.0f, &cx);
    printf("disjoint-clamp kernels -> %d (expect 0 DIVERGENT), cx=%.6f\n", r, cx);
    assert(r == 0);
    assert(cx >= 0.0f && cx <= 1.0f);

    /* Overlapping-but-different specs -> UNKNOWN (honest, not 'equal').
     * cand clamps to [0.1, 1.0]; output intervals overlap [0,1] vs [0.1,1]. */
    wubu_affine_clamp_t cand2 = {0.5f, 0.0f, 0.1f, 1.0f};
    r = wubu_kereq_prove_eq(&ref, &cand2, -3.0f, 3.0f, &cx);
    printf("overlapping specs -> %d (expect 2 UNKNOWN)\n", r);
    assert(r == 2);

    /* DA: reversed input range tolerated. */
    r = wubu_kereq_prove_eq(&ref, &same, 3.0f, -3.0f, &cx);
    printf("reversed range identical -> %d (expect 1)\n", r);
    assert(r == 1);

    printf("ALL KEREQ TESTS PASSED (genuine prover)\n");
    return 0;
}
