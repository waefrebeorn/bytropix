/* test_consolidate.c -- PATHS #320 memory consolidation (sleep replay).
 * Proves: salient memories strengthened by replay, weak unreplayed memories
 * pruned, strength grows with replay, live count, fail-closed on NULL. */
#include "wubu_consolidate.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_consolidate_t c;
    CHECK(wubu_consol_init(&c, 0.3) == 0, "init ok (threshold 0.3)");

    /* add memories: 2 salient, 3 weak */
    wubu_consol_add(&c, 1, 0.9);   /* salient */
    wubu_consol_add(&c, 2, 0.8);   /* salient */
    wubu_consol_add(&c, 3, 0.1);   /* weak */
    wubu_consol_add(&c, 4, 0.2);   /* weak */
    wubu_consol_add(&c, 5, 0.05);  /* weak */
    CHECK(wubu_consol_live_count(&c) == 5, "5 live memories");

    /* sleep: replay top-2 (the salient ones), prune the weak unreplayed */
    int pruned = wubu_consol_sleep(&c, 2);
    CHECK(pruned == 3, "3 weak memories pruned");
    CHECK(wubu_consol_live_count(&c) == 2, "2 salient memories survive");

    /* salient memories strengthened (strength > initial) */
    double s1 = wubu_consol_strength(&c, 1);
    CHECK(s1 > 0.9, "salient memory 1 strengthened (>0.9)");

    /* pruned memory strength = 0 */
    CHECK(wubu_consol_strength(&c, 3) == 0.0, "pruned memory strength 0");

    /* more sleep passes keep strengthening the salient ones */
    double before = wubu_consol_strength(&c, 1);
    wubu_consol_sleep(&c, 2);
    double after = wubu_consol_strength(&c, 1);
    CHECK(after >= before, "strength non-decreasing with more replay");

    /* salience clamps to [0,1] */
    wubu_consolidate_t d;
    wubu_consol_init(&d, 0.1);
    wubu_consol_add(&d, 10, 5.0);   /* over 1 */
    wubu_consol_add(&d, 11, -1.0);  /* under 0 */
    CHECK(wubu_consol_strength(&d, 10) <= 1.0, "salience clamped to <=1");

    /* fail-closed */
    CHECK(wubu_consol_sleep(NULL, 2) == -1, "NULL sleep -> -1");
    CHECK(wubu_consol_strength(NULL, 1) == 0.0, "NULL strength -> 0");
    CHECK(wubu_consol_live_count(NULL) == 0, "NULL live_count -> 0");
    CHECK(wubu_consol_add(NULL, 1, 0.5) == -1, "NULL add -> -1");

    printf("\n=== test_consolidate: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
