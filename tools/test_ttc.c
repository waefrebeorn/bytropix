/*
 * test_ttc.c -- Q08/Q15/Q20/R01/R03 verification.
 */
#include "wubu_ttc.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_ttc (Q08/Q15/Q20/R01/R03) ===\n");

    /* Q08 PolyKV coherence: identical summaries -> coherent; orthogonal -> not. */
    float a[2] = {1,0}, b[2] = {1,0}, c[2] = {0,1};
    CHECK(wubu_polykv_coherent(a, b, 2, 0.9f) == 1, "identical -> coherent");
    CHECK(wubu_polykv_coherent(a, c, 2, 0.9f) == 0, "orthogonal -> not coherent");

    /* Q15 HotPrefix: hot (freq 10, age 1) > cold (freq 1, age 100). */
    float hot = wubu_hotprefix_priority(10, 1, 0.1f);
    float cold = wubu_hotprefix_priority(1, 100, 0.1f);
    CHECK(hot > cold, "hot prefix prioritized over cold");

    /* Q20 budget steps: B=100, cost=7 -> 14 steps. */
    CHECK(wubu_ttc_budget_steps(100, 7.0f) == 14, "budget/steps = 14");
    CHECK(wubu_ttc_budget_steps(0, 7.0f) == 0, "zero budget -> 0");

    /* R01 scaling: q=1 -> smax; q=0 -> smin. */
    CHECK(fabs(wubu_scaling_factor(1.0f, 1.0f, 4.0f) - 4.0f) < 1e-5f, "q=1 -> smax");
    CHECK(fabs(wubu_scaling_factor(0.0f, 1.0f, 4.0f) - 1.0f) < 1e-5f, "q=0 -> smin");

    /* R03 CATTS: draft 10, conf 0.5 -> 5 tokens. */
    CHECK(wubu_catts_tokens(10, 0.5f) == 5, "draft 10, conf 0.5 -> 5");
    CHECK(wubu_catts_tokens(10, 0.0f) == 1, "conf 0 -> min 1");

    if (failures == 0) { printf("ALL TTC TESTS PASSED\n"); return 0; }
    printf("%d TTC TEST(S) FAILED\n", failures);
    return 1;
}
