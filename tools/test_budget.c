/* test_budget.c -- AGENT #161 resource budget bounds mutation search.
 * Proves: budget caps mutations + evals, exhaustion refuses (fail-closed),
 * remaining counts, fail-closed on NULL. */
#include "wubu_budget.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_budget_t b;
    CHECK(wubu_budget_init(&b, 3, 5) == 0, "init ok (3 mutations, 5 evals)");
    CHECK(wubu_budget_remaining_mutations(&b) == 3, "3 mutations remaining");
    CHECK(wubu_budget_remaining_evals(&b) == 5, "5 evals remaining");
    CHECK(wubu_budget_may_propose(&b) == 1, "may propose");
    CHECK(wubu_budget_exhausted(&b) == 0, "not exhausted");

    /* propose 3 mutations (the cap) */
    CHECK(wubu_budget_propose(&b) == 0, "propose 1");
    CHECK(wubu_budget_propose(&b) == 0, "propose 2");
    CHECK(wubu_budget_propose(&b) == 0, "propose 3");
    CHECK(wubu_budget_remaining_mutations(&b) == 0, "0 mutations remaining");
    CHECK(wubu_budget_may_propose(&b) == 0, "may NOT propose (cap hit)");
    CHECK(wubu_budget_propose(&b) == -1, "propose 4 -> -1 (fail-closed)");

    /* evals: 5 cap */
    for (int i = 0; i < 5; i++) CHECK(wubu_budget_eval(&b) == 0, "eval ok");
    CHECK(wubu_budget_may_eval(&b) == 0, "may NOT eval (cap hit)");
    CHECK(wubu_budget_eval(&b) == -1, "eval 6 -> -1 (fail-closed)");
    CHECK(wubu_budget_exhausted(&b) == 1, "fully exhausted");

    /* fail-closed on NULL */
    CHECK(wubu_budget_may_propose(NULL) == 0, "NULL may_propose -> 0");
    CHECK(wubu_budget_may_eval(NULL) == 0, "NULL may_eval -> 0");
    CHECK(wubu_budget_propose(NULL) == -1, "NULL propose -> -1");
    CHECK(wubu_budget_eval(NULL) == -1, "NULL eval -> -1");
    CHECK(wubu_budget_exhausted(NULL) == 1, "NULL exhausted -> 1 (fail-closed)");
    CHECK(wubu_budget_remaining_mutations(NULL) == 0, "NULL remaining_mut -> 0");
    CHECK(wubu_budget_remaining_evals(NULL) == 0, "NULL remaining_eval -> 0");

    printf("\n=== test_budget: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
