/*
 * test_orbit.c -- experts as gravitating bodies (WuBu STAR solar system).
 *
 * Verifies the properties the router depends on: a circular orbit stays
 * circular, potential ordering is monotone in radius, routing sends a token to
 * the deepest well, Kepler periods and 1:2 resonance are computed correctly,
 * cooling migrates bodies inward but never through the core floor.
 */
#include "wubu_orbit.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int fails = 0, checks = 0;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else printf("  [PASS] %s\n", m); } while (0)

int main(void)
{
    printf("=== test_orbit (solar-system object model) ===\n");

    /* 1. Circular orbit stability. */
    wubu_orbit_body b = { .id = 1, .mass = 1.0, .x = 0.30, .y = 0.0,
                          .temp = 4310.0 };
    b.vy = sqrt(1.0 / 0.30);   /* v = sqrt(GM/r), G=M=1 */
    wubu_orbit_sim s = { .g = 1.0, .central_mass = 1.0, .dt = 1e-3,
                         .softening = 1e-6, .disk_limit = 0.999 };
    double r0 = wubu_orbit_radius(&b);
    int escaped = 0;
    for (int i = 0; i < 5000; i++) escaped += wubu_orbit_step(&b, 1, &s);
    double r1 = wubu_orbit_radius(&b);
    printf("  radius %.6f -> %.6f (drift %.3f%%)\n", r0, r1,
           100.0 * fabs(r1 - r0) / r0);
    CHECK(fabs(r1 - r0) / r0 < 0.01, "circular orbit stays circular (<1% drift)");
    CHECK(escaped == 0, "no escape on a bound orbit");
    CHECK(b.temp == 4310.0, "identity temperature survives integration");

    /* 2. Potential is deeper closer in. */
    wubu_orbit_body inner = { .mass = 1.0, .x = 0.10, .y = 0.0 };
    wubu_orbit_body outer = { .mass = 1.0, .x = 0.50, .y = 0.0 };
    double pin = wubu_orbit_potential(&inner, 1.0, 1.0, 1e-6);
    double pout = wubu_orbit_potential(&outer, 1.0, 1.0, 1e-6);
    CHECK(pin < pout, "potential is deeper at smaller radius");
    CHECK(pin < 0.0 && pout < 0.0, "gravitational potential is negative");

    /* 3. Routing sends a token to the deepest well. */
    wubu_orbit_body bodies[2] = {
        { .id = 1, .mass = 1.0, .x = 0.10, .y = 0.0 },
        { .id = 2, .mass = 1.0, .x = 0.50, .y = 0.0 },
    };
    double w[2] = {0};
    CHECK(wubu_orbit_route(bodies, 2, 0.0, 0.0, 0.05, 1.0, 1e-3, w) == 0,
          "route runs");
    CHECK(w[0] > w[1], "token falls toward the inner (deeper) body");
    CHECK(fabs((w[0] + w[1]) - 1.0) < 1e-9, "routing weights sum to 1");

    /* 4. Kepler period. */
    wubu_orbit_body q = { .mass = 1.0, .x = 0.25, .y = 0.0 };
    double P = wubu_orbit_period(&q, 1.0, 1.0);
    double expect = 2.0 * M_PI * sqrt(0.25 * 0.25 * 0.25);
    printf("  period %.6f (expect %.6f)\n", P, expect);
    CHECK(fabs(P - expect) < 1e-9, "Kepler period matches 2*pi*sqrt(r^3/GM)");

    /* 5. 1:2 resonance: radius ratio 2^(2/3). */
    wubu_orbit_body a = { .mass = 1.0, .x = 0.25, .y = 0.0 };
    wubu_orbit_body c = { .mass = 1.0, .x = 0.25 * pow(2.0, 2.0 / 3.0), .y = 0.0 };
    CHECK(wubu_orbit_resonance(&a, &c, 1.0, 1.0, 2, 1, 1e-6),
          "1:2 resonance detected at the right radius");
    CHECK(!wubu_orbit_resonance(&a, &c, 1.0, 1.0, 2, 3, 1e-3),
          "non-resonant pair rejected");

    /* 6. Cooling migration respects the crystallized floor. */
    wubu_orbit_body m = { .mass = 1.0, .x = 0.50, .y = 0.0 };
    CHECK(wubu_orbit_cool(&m, 1, 0.10, 1.0, 0.0) == 1, "cool moves a body");
    CHECK(fabs(wubu_orbit_radius(&m) - 0.45) < 1e-9, "cool contracts by rate*dt");
    wubu_orbit_body f = { .mass = 1.0, .x = 0.06, .y = 0.0 };
    wubu_orbit_cool(&f, 1, 0.90, 10.0, 0.05);
    CHECK(fabs(wubu_orbit_radius(&f) - 0.05) < 1e-9, "cool never passes the core floor");

    /* 7. Poincare distance. */
    double d0 = wubu_orbit_poincare_dist(0.0, 0.0, 0.5, 0.0);
    double expect_d = acosh(1.0 + 2.0 * 0.25 / 0.75);
    printf("  poincare d(0,0.5) = %.6f (expect %.6f)\n", d0, expect_d);
    CHECK(fabs(d0 - expect_d) < 1e-9, "Poincare distance formula");
    CHECK(wubu_orbit_poincare_dist(0.2, 0.3, 0.2, 0.3) < 1e-12,
          "distance to self is zero");
    CHECK(wubu_orbit_poincare_dist(1.0, 0.0, 0.5, 0.0) == -1.0,
          "boundary point rejected");

    /* 8. Escape is counted and contained. */
    wubu_orbit_body e = { .mass = 1.0, .x = 0.2, .y = 0.0, .vy = 5.0 };
    int esc = 0;
    for (int i = 0; i < 2000; i++) esc += wubu_orbit_step(&e, 1, &s);
    CHECK(esc > 0, "unbound body reports escape");
    CHECK(wubu_orbit_radius(&e) <= 0.9991, "escaped body is clamped inside the disk");

    /* 9. Argument validation. */
    CHECK(wubu_orbit_route(NULL, 2, 0, 0, 0.1, 1, 1e-3, w) == -1, "NULL bodies -> -1");
    CHECK(wubu_orbit_route(bodies, 2, 0, 0, 0.0, 1, 1e-3, w) == -1, "zero temperature -> -1");
    CHECK(wubu_orbit_period(NULL, 1, 1) == -1.0, "NULL body -> -1 period");

    printf("\n=== test_orbit: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}