/* test_canary.c -- OMARCHY #286 rolling-release canary gate.
 * Proves: healthy canary -> PROMOTE, regressed canary -> ROLLBACK, too few
 * samples -> HOLD, fail-closed on NULL. */
#include "wubu_canary.h"
#include <stdio.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_canary_t c;
    /* canary may be at most 2x stable error rate; need 10 samples */
    CHECK(wubu_canary_init(&c, 2.0, 10) == 0, "init ok (2x ratio, 10 samples)");

    /* too few samples -> HOLD */
    wubu_canary_sample(&c, 0, 0);
    wubu_canary_sample(&c, 0, 0);
    CHECK(wubu_canary_decide(&c) == WUBU_CANARY_HOLD, "too few samples -> HOLD");

    /* healthy canary: both clean, 10 total samples */
    wubu_canary_t h;
    wubu_canary_init(&h, 2.0, 10);
    for (int i = 0; i < 10; i++) wubu_canary_sample(&h, 0, 0);
    CHECK(wubu_canary_decide(&h) == WUBU_CANARY_PROMOTE, "clean canary -> PROMOTE");

    /* canary slightly worse but within 2x of stable: stable 10% err,
     * canary 15% err -> 1.5x <= 2x -> PROMOTE */
    wubu_canary_t w;
    wubu_canary_init(&w, 2.0, 10);
    for (int i = 0; i < 10; i++) {
        wubu_canary_sample(&w, (i < 2) ? 1 : 0,   /* canary 20% err */
                                (i == 0) ? 1 : 0); /* stable 10% err */
    }
    /* canary 20% vs stable 10% = 2.0x, not > 2.0x -> PROMOTE */
    CHECK(wubu_canary_decide(&w) == WUBU_CANARY_PROMOTE, "2.0x (at limit) -> PROMOTE");

    /* regressed canary: stable clean, canary has errors -> ROLLBACK */
    wubu_canary_t bad;
    wubu_canary_init(&bad, 2.0, 10);
    for (int i = 0; i < 10; i++) {
        wubu_canary_sample(&bad, (i < 3) ? 1 : 0,   /* canary 30% err */
                                0);                   /* stable 0% err */
    }
    CHECK(wubu_canary_decide(&bad) == WUBU_CANARY_ROLLBACK, "errors on clean stable -> ROLLBACK");

    /* canary much worse than stable (stable 10%, canary 50%) -> ROLLBACK */
    wubu_canary_t bad2;
    wubu_canary_init(&bad2, 2.0, 10);
    for (int i = 0; i < 10; i++) {
        wubu_canary_sample(&bad2, (i < 5) ? 1 : 0,   /* canary 50% */
                                (i < 1) ? 1 : 0);     /* stable 10% */
    }
    CHECK(wubu_canary_decide(&bad2) == WUBU_CANARY_ROLLBACK, "5x stable -> ROLLBACK");

    /* fail-closed on NULL */
    CHECK(wubu_canary_decide(NULL) == WUBU_CANARY_ROLLBACK, "NULL decide -> ROLLBACK (fail-closed)");
    CHECK(wubu_canary_sample(NULL, 0, 0) == -1, "NULL sample -> -1");
    CHECK(wubu_canary_err_rate(NULL) == 0.0, "NULL err_rate -> 0");

    printf("\n=== test_canary: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
