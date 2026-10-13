/* test_roadmap.c -- PATHS #329 AGI roadmap milestone tracker.
 * Proves: add/complete, progress, overdue detection, health signal,
 * fail-closed on NULL. */
#include "wubu_roadmap.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_roadmap_t r;
    /* current quarter 2026Q4 = 20264 */
    CHECK(wubu_rm_init(&r, 20264) == 0, "init ok (current 2026Q4)");

    /* milestones: some past target (overdue if not done), some future */
    wubu_rm_add(&r, "compiler-selfhost", 20263, WUBU_RM_DONE);         /* done, was due Q3 */
    wubu_rm_add(&r, "kernel-9p", 20263, WUBU_RM_DONE);                 /* done */
    wubu_rm_add(&r, "jit-recursion", 20263, WUBU_RM_IN_PROGRESS);      /* OVERDUE */
    wubu_rm_add(&r, "lean-proofs", 20264, WUBU_RM_NOT_STARTED);        /* due now, not overdue yet */
    wubu_rm_add(&r, "agi-demo", 20271, WUBU_RM_NOT_STARTED);           /* future */
    CHECK(r.n == 5, "5 milestones");

    /* progress: 2 done / 5 = 0.4 */
    CHECK(wubu_rm_progress(&r) > 0.39 && wubu_rm_progress(&r) < 0.41, "progress = 0.4 (2/5)");

    /* overdue: jit-recursion (Q3 < Q4, not done) = 1 */
    CHECK(wubu_rm_overdue(&r) == 1, "1 overdue (jit-recursion)");

    /* health: 1 overdue of 5 = 20% <= 25% -> at-risk */
    CHECK(wubu_rm_health(&r) == 1, "health at-risk (1 overdue)");

    /* complete the overdue one -> no overdue -> on-track */
    CHECK(wubu_rm_complete(&r, "jit-recursion") == 0, "complete jit-recursion");
    CHECK(wubu_rm_overdue(&r) == 0, "0 overdue now");
    CHECK(wubu_rm_health(&r) == 0, "health on-track");
    CHECK(wubu_rm_progress(&r) > 0.59 && wubu_rm_progress(&r) < 0.61, "progress = 0.6 (3/5)");

    /* advance time: now lean-proofs (Q4) and more become overdue */
    wubu_roadmap_t r2;
    wubu_rm_init(&r2, 20282);   /* 2028Q2, well past all targets */
    wubu_rm_add(&r2, "a", 20263, WUBU_RM_IN_PROGRESS);
    wubu_rm_add(&r2, "b", 20264, WUBU_RM_NOT_STARTED);
    wubu_rm_add(&r2, "c", 20271, WUBU_RM_DONE);
    CHECK(wubu_rm_overdue(&r2) == 2, "2 overdue (a,b); c is done");
    /* 2 of 3 = 66% > 25% -> behind */
    CHECK(wubu_rm_health(&r2) == 2, "health behind (66% overdue)");

    /* fail-closed */
    CHECK(wubu_rm_init(NULL, 20264) == -1, "NULL init -> -1");
    CHECK(wubu_rm_add(&r, NULL, 1, WUBU_RM_DONE) == -1, "NULL name -> -1");
    CHECK(wubu_rm_add(&r, "x", 1, 99) == -1, "bad status -> -1");
    CHECK(wubu_rm_complete(&r, "nope") == -1, "complete unknown -> -1");
    CHECK(wubu_rm_complete(NULL, "x") == -1, "NULL complete -> -1");
    CHECK(wubu_rm_progress(NULL) == 0.0, "NULL progress -> 0");
    CHECK(wubu_rm_overdue(NULL) == 0, "NULL overdue -> 0");
    CHECK(wubu_rm_health(NULL) == 0, "NULL health -> 0");

    printf("\n=== test_roadmap: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
