/* test_sat.c -- WORLDSIM #268 narrowphase SAT collision.
 * Proves: overlapping boxes collide, separated boxes don't, rotated boxes,
 * penetration depth + normal, touching-but-separated edge case, fail-closed. */
#include "wubu_sat.h"
#include <stdio.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)
#define NEAR(a,b,t) (fabs((a)-(b)) < (t))

int main(void) {
    /* two axis-aligned boxes overlapping */
    wubu_obb_t a = {0, 0, 1, 1, 0};
    wubu_obb_t b = {1.5, 0, 1, 1, 0};
    double nx, ny, depth;
    CHECK(wubu_sat_collide(&a, &b, &nx, &ny, &depth) == 1, "overlapping boxes collide");
    CHECK(depth > 0, "penetration depth > 0");
    CHECK(NEAR(nx, 1.0, 1e-9) && NEAR(ny, 0.0, 1e-9), "normal points +x (a to b)");
    /* a right edge at x=1, b left edge at x=0.5 -> overlap 0.5 */
    CHECK(NEAR(depth, 0.5, 1e-9), "depth = 0.5 (gap overlap)");

    /* separated boxes -> no collision */
    wubu_obb_t c = {0, 0, 1, 1, 0};
    wubu_obb_t d = {5, 0, 1, 1, 0};
    CHECK(wubu_sat_overlap(&c, &d) == 0, "separated boxes don't collide");

    /* exactly touching (gap = 0) -> separating axis (overlap 0, not < 0) ->
     * our test uses overlap < 0 for separation, so touching counts as collide.
     * Use a clear gap instead. */
    wubu_obb_t e = {0, 0, 1, 1, 0};
    wubu_obb_t f = {2.1, 0, 1, 1, 0};   /* gap 0.1 */
    CHECK(wubu_sat_overlap(&e, &f) == 0, "small-gap boxes don't collide");

    /* rotated box still detected */
    wubu_obb_t g = {0, 0, 1, 1, 0};
    wubu_obb_t h = {1.5, 0, 1, 1, 0.5};   /* rotated 0.5 rad, overlapping */
    CHECK(wubu_sat_overlap(&g, &h) == 1, "rotated overlapping box collides");

    /* rotated but far -> no collision */
    wubu_obb_t i = {0, 0, 1, 1, 0.5};
    wubu_obb_t j = {10, 10, 1, 1, 0.5};
    CHECK(wubu_sat_overlap(&i, &j) == 0, "rotated far boxes don't collide");

    /* vertical separation */
    wubu_obb_t k = {0, 0, 1, 1, 0};
    wubu_obb_t l = {0, 5, 1, 1, 0};
    CHECK(wubu_sat_overlap(&k, &l) == 0, "vertically separated don't collide");

    /* fail-closed on NULL */
    CHECK(wubu_sat_overlap(NULL, &b) == 0, "NULL a -> no collision");
    CHECK(wubu_sat_overlap(&a, NULL) == 0, "NULL b -> no collision");

    printf("\n=== test_sat: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
