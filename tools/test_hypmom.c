/* test_hypmom.c -- WORLDSIM #270 hyperbolic momentum embedder.
 * Proves: exp map keeps points inside the disk, momentum step moves the point,
 * hyperbolic distance >= 0 and d(x,x)=0, distance grows toward the boundary,
 * fail-closed on NULL/bad params. */
#include "wubu_hypmom.h"
#include <stdio.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_hypmom_t h;
    CHECK(wubu_hm_init(&h, -1.0) == 0, "init ok (curvature -1)");
    CHECK(wubu_hm_init(&h, 1.0) == 0, "positive curvature coerced to -1");
    CHECK(h.curvature == -1.0, "curvature is -1");

    /* exp map from origin with a velocity -> a point inside the disk */
    double v[WUBU_HM_DIM] = {0.1, 0, 0, 0, 0, 0, 0, 0};
    double out[WUBU_HM_DIM];
    CHECK(wubu_hm_exp_map(&h, v, out) == 0, "exp map ok");
    double n = wubu_hm_norm(out, WUBU_HM_DIM);
    CHECK(n < 1.0, "exp-mapped point is inside the disk");

    /* zero velocity -> stays at the origin */
    double zero[WUBU_HM_DIM] = {0};
    double out0[WUBU_HM_DIM];
    wubu_hm_exp_map(&h, zero, out0);
    CHECK(wubu_hm_norm(out0, WUBU_HM_DIM) == 0.0, "zero velocity -> stays at origin");

    /* momentum step moves the point toward the gradient */
    double g[WUBU_HM_DIM] = {0.5, 0.3, 0, 0, 0, 0, 0, 0};
    double before = wubu_hm_norm(h.p, WUBU_HM_DIM);
    CHECK(wubu_hm_step(&h, g, 0.1, 0.9) == 0, "momentum step ok");
    double after = wubu_hm_norm(h.p, WUBU_HM_DIM);
    CHECK(after > before, "point moved away from origin");
    CHECK(after < 1.0, "point still inside the disk");

    /* hyperbolic distance: d(x,x) = 0 */
    double a[WUBU_HM_DIM] = {0.1, 0.2, 0, 0, 0, 0, 0, 0};
    double b[WUBU_HM_DIM] = {0.1, 0.2, 0, 0, 0, 0, 0, 0};
    CHECK(wubu_hm_distance(a, b, -1.0) < 1e-9, "d(x,x) = 0");

    /* distance is non-negative and symmetric */
    double c[WUBU_HM_DIM] = {0.3, -0.2, 0.1, 0, 0, 0, 0, 0};
    double d1 = wubu_hm_distance(a, c, -1.0);
    double d2 = wubu_hm_distance(c, a, -1.0);
    CHECK(d1 >= 0.0, "distance >= 0");
    CHECK(fabs(d1 - d2) < 1e-9, "distance symmetric");
    CHECK(d1 > 0.0, "distinct points have positive distance");

    /* distance grows toward the boundary (a point near the edge is far from origin) */
    double edge[WUBU_HM_DIM] = {0.95, 0, 0, 0, 0, 0, 0, 0};
    double origin[WUBU_HM_DIM] = {0};
    double d_edge = wubu_hm_distance(origin, edge, -1.0);
    CHECK(d_edge > d1, "boundary point is farther from origin than interior point");

    /* fail-closed */
    CHECK(wubu_hm_exp_map(NULL, v, out) == -1, "NULL exp_map -> -1");
    CHECK(wubu_hm_step(&h, g, -0.1, 0.9) == -1, "negative lr -> -1");
    CHECK(wubu_hm_step(&h, g, 0.1, 2.0) == -1, "momentum > 1 -> -1");
    CHECK(wubu_hm_distance(NULL, a, -1.0) == 0.0, "NULL distance -> 0");
    CHECK(wubu_hm_norm(NULL, 8) == 0.0, "NULL norm -> 0");

    printf("\n=== test_hypmom: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
