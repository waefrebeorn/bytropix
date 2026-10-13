/* test_lockdep.c -- KERNEL #24 lockdep-style deadlock detector.
 * Proves: consistent order ok, ABBA inversion detected, violation count,
 * last-violation pair, fail-closed on NULL. */
#include "wubu_lockdep.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_lockdep_t ld;
    CHECK(wubu_ld_init(&ld, 4) == 0, "init ok (4 locks)");

    /* consistent ordering: task1 acquires A then B (records A->B) */
    CHECK(wubu_ld_acquire(&ld, 0) == 0, "acquire A");
    CHECK(wubu_ld_acquire(&ld, 1) == 0, "acquire B (A->B recorded)");
    wubu_ld_release(&ld);
    wubu_ld_release(&ld);
    CHECK(wubu_ld_violations(&ld) == 0, "no violation for consistent order");

    /* consistent again: A then B -> fine */
    CHECK(wubu_ld_acquire(&ld, 0) == 0, "acquire A again");
    CHECK(wubu_ld_acquire(&ld, 1) == 0, "acquire B again (ok)");
    wubu_ld_release(&ld);
    wubu_ld_release(&ld);

    /* ABBA inversion: acquire B then A -> B->A, but A->B already observed */
    CHECK(wubu_ld_acquire(&ld, 1) == 0, "acquire B");
    int r = wubu_ld_acquire(&ld, 0);   /* B held, acquiring A: order[A][B] set -> violation */
    CHECK(r == 1, "acquiring A while holding B is an ABBA violation");
    CHECK(wubu_ld_violations(&ld) == 1, "1 violation recorded");

    int a, b;
    CHECK(wubu_ld_last_violation(&ld, &a, &b) == 0, "last violation available");
    CHECK(a == 1 && b == 0, "violation pair = (B=1, A=0)");

    /* a third distinct order that's also an inversion adds another violation */
    /* reset: release all */
    wubu_ld_release(&ld); wubu_ld_release(&ld);
    /* C then D (C->D), then D then C (inversion) */
    wubu_ld_acquire(&ld, 2); wubu_ld_acquire(&ld, 3);
    wubu_ld_release(&ld); wubu_ld_release(&ld);
    wubu_ld_acquire(&ld, 3);
    CHECK(wubu_ld_acquire(&ld, 2) == 1, "D then C is a violation (C->D observed)");
    CHECK(wubu_ld_violations(&ld) == 2, "2 violations total");

    /* fail-closed */
    CHECK(wubu_ld_acquire(NULL, 0) == -1, "NULL acquire -> -1");
    CHECK(wubu_ld_acquire(&ld, 99) == -1, "out-of-range lock -> -1");
    CHECK(wubu_ld_release(NULL) == -1, "NULL release -> -1");
    CHECK(wubu_ld_violations(NULL) == 0, "NULL violations -> 0");
    CHECK(wubu_ld_last_violation(NULL, &a, &b) == -1, "NULL last_violation -> -1");
    CHECK(wubu_ld_init(NULL, 4) == -1, "NULL init -> -1");

    printf("\n=== test_lockdep: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
