/* test_humangate.c -- AGENT #146 DA-3 human gate + freeze (death condition).
 * Proves: unfrozen allows mutation, human veto freezes (denies mutation),
 * one-way to set, wrong-token unfreeze refused, correct-token unfreeze works,
 * veto count, fail-closed on NULL. */
#include "wubu_humangate.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_humangate_t g;
    uint64_t TOKEN = 0xCAFEBABE;
    CHECK(wubu_humangate_init(&g, TOKEN) == 0, "init ok");
    CHECK(wubu_humangate_frozen(&g) == 0, "starts unfrozen");
    CHECK(wubu_humangate_may_mutate(&g) == 1, "unfrozen: mutation allowed");

    /* human veto -> freeze */
    CHECK(wubu_humangate_veto(&g) == 0, "veto ok");
    CHECK(wubu_humangate_frozen(&g) == 1, "frozen after veto");
    CHECK(wubu_humangate_may_mutate(&g) == 0, "frozen: mutation DENIED (fail-closed)");
    CHECK(wubu_humangate_vetoes(&g) == 1, "veto count 1");

    /* one-way: the loop cannot clear the freeze itself (no unfreeze w/o token) */
    CHECK(wubu_humangate_unfreeze(&g, 0xDEADBEEF) == 0, "wrong token: unfreeze refused");
    CHECK(wubu_humangate_frozen(&g) == 1, "stays frozen after wrong token");
    CHECK(wubu_humangate_may_mutate(&g) == 0, "still denying after wrong token");

    /* correct token unfreezes */
    CHECK(wubu_humangate_unfreeze(&g, TOKEN) == 1, "correct token: unfrozen");
    CHECK(wubu_humangate_frozen(&g) == 0, "unfrozen");
    CHECK(wubu_humangate_may_mutate(&g) == 1, "mutation allowed again");

    /* re-veto increments count */
    wubu_humangate_veto(&g);
    CHECK(wubu_humangate_vetoes(&g) == 2, "veto count 2 after re-veto");

    /* fail-closed on NULL */
    CHECK(wubu_humangate_may_mutate(NULL) == 0, "NULL may_mutate -> 0 (deny)");
    CHECK(wubu_humangate_frozen(NULL) == 1, "NULL frozen -> 1 (fail-closed)");
    CHECK(wubu_humangate_veto(NULL) == -1, "NULL veto -> -1");
    CHECK(wubu_humangate_unfreeze(NULL, TOKEN) == 0, "NULL unfreeze -> 0");
    CHECK(wubu_humangate_vetoes(NULL) == 0, "NULL vetoes -> 0");

    printf("\n=== test_humangate: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
