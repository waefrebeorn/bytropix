/* test_failweight.c -- AGENT #148 DA-2 failure weighting 3x bounds search.
 * Proves: failures weighted 3x, high-failure region avoided, best region is
 * the low-penalty one, search bounds away from failing regions, fail-closed. */
#include "wubu_failweight.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_failweight_t f;
    CHECK(wubu_failweight_init(&f, 10, 3) == 0, "init 3x ok");
    CHECK(wubu_failweight_init(&f, 0, 3) == -1, "zero regions rejected");
    CHECK(wubu_failweight_init(&f, 99999, 3) == -1, "oversize rejected");

    /* region 0: all successes; region 1: all failures */
    for (int i = 0; i < 5; i++) wubu_failweight_record(&f, 0, 1);
    for (int i = 0; i < 5; i++) wubu_failweight_record(&f, 1, 0);

    /* 3x weighting: region 1 penalty = 5*3 = 15; region 0 = -5 */
    CHECK(wubu_failweight_penalty(&f, 1) == 15, "region 1 penalty = 5 failures * 3 = 15");
    CHECK(wubu_failweight_penalty(&f, 0) == -5, "region 0 penalty = 5 successes = -5");

    /* failing region avoided, succeeding region not */
    CHECK(wubu_failweight_avoid(&f, 1) == 1, "failing region avoided");
    CHECK(wubu_failweight_avoid(&f, 0) == 0, "succeeding region not avoided");

    /* best region = the succeeding one */
    CHECK(wubu_failweight_best(&f) == 0, "best region = 0 (lowest penalty)");

    /* 3x dominates: even 1 failure outweighs 2 successes (3 > 2) */
    wubu_failweight_t g;
    wubu_failweight_init(&g, 2, 3);
    wubu_failweight_record(&g, 0, 1); wubu_failweight_record(&g, 0, 1);  /* 2 succ */
    wubu_failweight_record(&g, 1, 0);                                     /* 1 fail */
    CHECK(wubu_failweight_penalty(&g, 1) == 3, "1 failure -> penalty 3");
    CHECK(wubu_failweight_penalty(&g, 0) == -2, "2 successes -> penalty -2");
    CHECK(wubu_failweight_avoid(&g, 1) == 1, "1 failure (3) > 2 successes (2): avoid");

    /* balanced region (1 fail 3 succ) not avoided: 3 - 3 = 0 */
    wubu_failweight_t h;
    wubu_failweight_init(&h, 1, 3);
    wubu_failweight_record(&h, 0, 0);
    wubu_failweight_record(&h, 0, 1); wubu_failweight_record(&h, 0, 1); wubu_failweight_record(&h, 0, 1);
    CHECK(wubu_failweight_penalty(&h, 0) == 0, "1 fail + 3 succ -> penalty 0 (balanced)");
    CHECK(wubu_failweight_avoid(&h, 0) == 0, "balanced region not avoided");

    /* fail-closed */
    CHECK(wubu_failweight_penalty(&f, 999) == 0, "out-of-range penalty -> 0");
    CHECK(wubu_failweight_avoid(&f, 999) == 1, "out-of-range avoid -> 1 (fail-safe)");
    CHECK(wubu_failweight_record(&f, 999, 1) == -1, "out-of-range record -> -1");
    CHECK(wubu_failweight_record(NULL, 0, 1) == -1, "NULL record -> -1");
    CHECK(wubu_failweight_best(NULL) == -1, "NULL best -> -1");

    printf("\n=== test_failweight: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
