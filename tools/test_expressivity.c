/* test_expressivity.c -- PATHS #324 expressivity-invariant risk tracking.
 * Proves: expressivity preserved -> SAFE, increased -> NOVEL, decreased ->
 * RISKY, baseline update on accept, risk rate, fail-closed on NULL. */
#include "wubu_expressivity.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_expressivity_t x;
    CHECK(wubu_expr_init(&x, 100.0, 0.05) == 0, "init ok (baseline 100, tol 5%)");

    /* expressivity preserved (within tolerance) -> SAFE */
    CHECK(wubu_expr_assess(&x, 102.0) == WUBU_EXPR_SAFE, "102 (within 5%) -> SAFE");
    CHECK(wubu_expr_assess(&x, 96.0) == WUBU_EXPR_SAFE, "96 (within 5%) -> SAFE");

    /* expressivity increased -> NOVEL */
    CHECK(wubu_expr_assess(&x, 120.0) == WUBU_EXPR_NOVEL, "120 (>5% up) -> NOVEL");

    /* expressivity decreased -> RISKY */
    CHECK(wubu_expr_assess(&x, 80.0) == WUBU_EXPR_RISKY, "80 (>5% down) -> RISKY");

    /* just outside tolerance on the downside -> RISKY */
    CHECK(wubu_expr_assess(&x, 94.0) == WUBU_EXPR_RISKY, "94 (just below 95) -> RISKY");

    /* accept a novel mutation -> baseline updates */
    wubu_expr_accept(&x, 120.0);
    /* now 120 is baseline; 118 is within 5% -> SAFE */
    CHECK(wubu_expr_assess(&x, 118.0) == WUBU_EXPR_SAFE, "after accept, 118 vs new baseline -> SAFE");

    /* risk rate */
    /* mutations so far: 102(S),96(S),120(N),80(R),94(R),118(S) = 6 total, 2 risky */
    CHECK(x.n_mutations == 6, "6 mutations assessed");
    CHECK(x.n_risky == 2, "2 risky");
    double rate = wubu_expr_risk_rate(&x);
    CHECK(rate > 0.3 && rate < 0.34, "risk rate ~ 2/6 = 0.33");

    /* fail-closed */
    CHECK(wubu_expr_assess(NULL, 100.0) == WUBU_EXPR_RISKY, "NULL assess -> RISKY (fail-closed)");
    CHECK(wubu_expr_accept(NULL, 100.0) == -1, "NULL accept -> -1");
    CHECK(wubu_expr_risk_rate(NULL) == 0.0, "NULL risk_rate -> 0");
    CHECK(wubu_expr_init(NULL, 1, 1) == -1, "NULL init -> -1");

    printf("\n=== test_expressivity: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
