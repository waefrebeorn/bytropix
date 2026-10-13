/* test_fitness_land.c -- AGENT #162 mutation fitness landscape telemetry.
 * Proves: best tracking, mean/variance, plateau detection, improvement resets
 * stagnation, fail-closed on NULL. */
#include "wubu_fitness_land.h"
#include <stdio.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)
#define NEAR(a,b,t) (fabs((a)-(b)) < (t))

int main(void) {
    wubu_fitness_land_t l;
    CHECK(wubu_fl_init(&l, 3) == 0, "init ok (plateau threshold 3)");

    /* improving sequence: 1, 2, 3, 5 */
    double pos[4] = {0,0,0,0};
    wubu_fl_record(&l, 1.0, pos);
    wubu_fl_record(&l, 2.0, pos);
    wubu_fl_record(&l, 3.0, pos);
    wubu_fl_record(&l, 5.0, pos);
    CHECK(NEAR(wubu_fl_best(&l), 5.0, 1e-9), "best = 5");
    CHECK(NEAR(wubu_fl_mean(&l), (1+2+3+5)/4.0, 1e-9), "mean = 2.75");
    CHECK(l.stagnant == 0, "no stagnation (always improving)");
    CHECK(wubu_fl_plateaued(&l) == 0, "not plateaued");

    /* non-improving sequence: 5, 4, 4, 4 (3 stagnant) -> plateau */
    wubu_fl_record(&l, 4.0, pos);
    wubu_fl_record(&l, 4.0, pos);
    wubu_fl_record(&l, 4.0, pos);
    CHECK(l.stagnant == 3, "3 stagnant steps");
    CHECK(wubu_fl_plateaued(&l) == 1, "plateaued (>= 3 stagnant)");
    CHECK(NEAR(wubu_fl_best(&l), 5.0, 1e-9), "best still 5");

    /* a new improvement resets stagnation */
    wubu_fl_record(&l, 6.0, pos);
    CHECK(l.stagnant == 0, "improvement resets stagnation");
    CHECK(NEAR(wubu_fl_best(&l), 6.0, 1e-9), "best updated to 6");

    /* variance: constant fitness -> 0 variance */
    wubu_fitness_land_t c;
    wubu_fl_init(&c, 3);
    wubu_fl_record(&c, 5.0, pos);
    wubu_fl_record(&c, 5.0, pos);
    wubu_fl_record(&c, 5.0, pos);
    CHECK(NEAR(wubu_fl_variance(&c), 0.0, 1e-9), "constant fitness -> 0 variance");

    /* NULL position handled */
    CHECK(wubu_fl_record(&c, 7.0, NULL) == 0, "NULL position accepted (defaults to 0)");

    /* fail-closed */
    CHECK(wubu_fl_record(NULL, 1.0, pos) == -1, "NULL record -> -1");
    CHECK(wubu_fl_mean(NULL) == 0.0, "NULL mean -> 0");
    CHECK(wubu_fl_variance(NULL) == 0.0, "NULL variance -> 0");
    CHECK(wubu_fl_best(NULL) == 0.0, "NULL best -> 0");
    CHECK(wubu_fl_plateaued(NULL) == 0, "NULL plateaued -> 0");

    printf("\n=== test_fitness_land: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
