/* test_islands.c -- AGENT #163 island-model parallel evolution.
 * Proves: islands evolve independently, migration every N generations spreads
 * the best, global best across islands, migration count, fail-closed on NULL. */
#include "wubu_islands.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_islands_t m;
    CHECK(wubu_isl_init(&m, 4, 3) == 0, "init ok (4 islands, migrate every 3 gens)");
    CHECK(wubu_isl_init(&m, 0, 3) == -1, "zero islands rejected");
    CHECK(wubu_isl_init(&m, 99, 3) == -1, "oversize islands rejected");

    /* seed islands with different bests */
    wubu_isl_add(&m, 0, 1.0); wubu_isl_add(&m, 0, 2.0);   /* island 0 best 2 */
    wubu_isl_add(&m, 1, 5.0);                             /* island 1 best 5 */
    wubu_isl_add(&m, 2, 3.0);                             /* island 2 best 3 */
    wubu_isl_add(&m, 3, 1.5);                             /* island 3 best 1.5 */

    /* island bests */
    CHECK(wubu_isl_island_best(&m, 0) == 2.0, "island 0 best = 2");
    CHECK(wubu_isl_island_best(&m, 1) == 5.0, "island 1 best = 5");
    /* global best */
    CHECK(wubu_isl_global_best(&m) == 5.0, "global best = 5");

    /* generations 1, 2: no migration yet */
    CHECK(wubu_isl_generation(&m) == 0, "gen 1: no migration");
    CHECK(wubu_isl_generation(&m) == 0, "gen 2: no migration");
    CHECK(wubu_isl_migrations(&m) == 0, "0 migrations so far");

    /* generation 3: migration fires (ring: each island's best -> next) */
    int migrated = wubu_isl_generation(&m);
    CHECK(migrated == 4, "gen 3: 4 migrations (one per island)");
    CHECK(wubu_isl_migrations(&m) == 4, "total 4 migrations");

    /* migration spread island 1's best (5) to island 2 */
    CHECK(wubu_isl_island_best(&m, 2) == 5.0, "island 2 received island 1's best (5)");

    /* generations 4, 5: no migration; gen 6: migration again */
    wubu_isl_generation(&m);
    wubu_isl_generation(&m);
    CHECK(wubu_isl_generation(&m) == 4, "gen 6: 4 more migrations");
    CHECK(wubu_isl_migrations(&m) == 8, "total 8 migrations");

    /* fail-closed */
    CHECK(wubu_isl_add(&m, 99, 1.0) == -1, "bad island -> -1");
    CHECK(wubu_isl_add(NULL, 0, 1.0) == -1, "NULL add -> -1");
    CHECK(wubu_isl_generation(NULL) == -1, "NULL generation -> -1");
    CHECK(wubu_isl_island_best(&m, 99) == 0.0, "bad island best -> 0");
    CHECK(wubu_isl_island_best(NULL, 0) == 0.0, "NULL island best -> 0");
    CHECK(wubu_isl_global_best(NULL) == 0.0, "NULL global best -> 0");

    printf("\n=== test_islands: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
