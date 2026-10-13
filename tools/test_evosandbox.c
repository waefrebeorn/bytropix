/* test_evosandbox.c -- PATHS #325 self-referential code evolution sandbox.
 * Proves: safe+improving mutation kept, unsafe reverted, non-improving
 * reverted, budget bounds mutations, fitness only moves on kept, fail-closed. */
#include "wubu_evosandbox.h"
#include <stdio.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_evosandbox_t s;
    CHECK(wubu_es_init(&s, 1.0, 5) == 0, "init ok (fitness 1.0, budget 5)");
    CHECK(wubu_es_fitness(&s) == 1.0, "fitness starts 1.0");
    CHECK(wubu_es_remaining(&s) == 5, "5 steps remaining");

    /* safe + improving -> kept, fitness improves */
    CHECK(wubu_es_mutate(&s, "m1", 2.0, 1) == 1, "safe+improving m1 kept");
    CHECK(wubu_es_fitness(&s) == 2.0, "fitness now 2.0");
    CHECK(wubu_es_remaining(&s) == 4, "4 steps remaining");

    /* safe but NOT improving -> reverted, fitness unchanged */
    CHECK(wubu_es_mutate(&s, "m2", 1.5, 1) == 0, "safe+non-improving m2 reverted");
    CHECK(wubu_es_fitness(&s) == 2.0, "fitness unchanged after revert");

    /* improving but UNSAFE -> reverted (safety wins over fitness) */
    CHECK(wubu_es_mutate(&s, "m3", 5.0, 0) == 0, "unsafe m3 reverted even though it improves");
    CHECK(wubu_es_fitness(&s) == 2.0, "fitness unchanged (unsafe mutation rejected)");

    /* safe + improving again -> kept */
    CHECK(wubu_es_mutate(&s, "m4", 3.0, 1) == 1, "safe+improving m4 kept");
    CHECK(wubu_es_fitness(&s) == 3.0, "fitness now 3.0");
    CHECK(wubu_es_remaining(&s) == 1, "1 step remaining");

    /* exhaust the budget */
    CHECK(wubu_es_mutate(&s, "m5", 4.0, 1) == 1, "m5 kept");
    CHECK(wubu_es_remaining(&s) == 0, "0 steps remaining");
    /* budget exhausted -> -1 */
    CHECK(wubu_es_mutate(&s, "m6", 9.0, 1) == -1, "budget exhausted -> -1");
    CHECK(wubu_es_fitness(&s) == 4.0, "fitness capped at last kept (4.0)");

    /* stats: kept m1,m4,m5 = 3; reverted m2,m3 = 2 */
    int kept, reverted;
    wubu_es_stats(&s, &kept, &reverted);
    CHECK(kept == 3, "3 kept");
    CHECK(reverted == 2, "2 reverted");

    /* fail-closed */
    CHECK(wubu_es_init(NULL, 1.0, 5) == -1, "NULL init -> -1");
    CHECK(wubu_es_init(&s, 1.0, -1) == -1, "negative budget -> -1");
    CHECK(wubu_es_mutate(&s, NULL, 2.0, 1) == -1, "NULL name -> -1");
    CHECK(wubu_es_mutate(NULL, "x", 2.0, 1) == -1, "NULL sandbox -> -1");
    CHECK(wubu_es_fitness(NULL) == 0.0, "NULL fitness -> 0");
    CHECK(wubu_es_remaining(NULL) == 0, "NULL remaining -> 0");

    printf("\n=== test_evosandbox: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
