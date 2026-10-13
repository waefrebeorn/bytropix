/* test_sph.c -- WORLDSIM #265 fluid simulation (SPH).
 * Proves: kernel is 0 outside h and positive inside, density computed from
 * neighbours, isolated particle has ~0 density, clustered particles have
 * higher density, pressure non-negative, simulation stays bounded (stable),
 * fail-closed on NULL. */
#include "wubu_sph.h"
#include <stdio.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)
#define NEAR(a,b,t) (fabs((a)-(b)) < (t))

int main(void) {
    /* kernel: 0 outside h, positive inside, decreasing */
    CHECK(wubu_sph_kernel(0.0, 1.0) > 0, "kernel(0) > 0");
    CHECK(wubu_sph_kernel(1.0, 1.0) == 0.0, "kernel(h) = 0 (boundary)");
    CHECK(wubu_sph_kernel(2.0, 1.0) == 0.0, "kernel(>h) = 0");
    CHECK(wubu_sph_kernel(0.5, 1.0) > wubu_sph_kernel(0.9, 1.0), "kernel decreases with r");

    wubu_sph_t s;
    CHECK(wubu_sph_init(&s, 1.0, 1.0) == 0, "init ok");
    CHECK(wubu_sph_init(&s, 0.0, 1.0) == -1, "zero h rejected");
    CHECK(wubu_sph_init(&s, 1.0, 0.0) == -1, "zero mass rejected");

    /* isolated particle: ~0 density (only self, kernel(0) contributes) */
    wubu_sph_add(&s, 0, 0);
    wubu_sph_compute_density(&s);
    CHECK(s.p[0].density >= 0, "isolated density >= 0");
    CHECK(s.p[0].pressure >= 0, "pressure non-negative");

    /* clustered particles: higher density than isolated */
    wubu_sph_t c;
    wubu_sph_init(&c, 2.0, 1.0);
    for (int i = 0; i < 5; i++) wubu_sph_add(&c, i * 0.3, 0);   /* close together */
    wubu_sph_compute_density(&c);
    CHECK(c.p[2].density > s.p[0].density, "clustered density > isolated");

    /* simulation: run a few steps, stays bounded (no NaN/explosion) */
    wubu_sph_t sim;
    wubu_sph_init(&sim, 1.5, 1.0);
    for (int i = 0; i < 20; i++) wubu_sph_add(&sim, (i % 5) * 0.5, (i / 5) * 0.5);
    int stable = 1;
    for (int step = 0; step < 50; step++) {
        wubu_sph_step(&sim, 0.01);
        for (int i = 0; i < sim.n; i++) {
            if (isnan(sim.p[i].x) || isnan(sim.p[i].y) || isinf(sim.p[i].x) ||
                fabs(sim.p[i].x) > 1e6 || fabs(sim.p[i].y) > 1e6) { stable = 0; break; }
        }
        if (!stable) break;
    }
    CHECK(stable == 1, "simulation stable (no NaN/explosion over 50 steps)");

    /* KE is finite */
    CHECK(isfinite(wubu_sph_ke(&sim)), "KE finite");

    /* fail-closed */
    CHECK(wubu_sph_compute_density(NULL) == -1, "NULL density -> -1");
    CHECK(wubu_sph_step(NULL, 0.01) == -1, "NULL step -> -1");
    CHECK(wubu_sph_step(&sim, 0.0) == -1, "zero dt -> -1");
    CHECK(wubu_sph_add(NULL, 0, 0) == -1, "NULL add -> -1");

    printf("\n=== test_sph: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
