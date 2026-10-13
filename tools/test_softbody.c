/* test_softbody.c -- WORLDSIM #266 soft-body dynamics.
 * Proves: pinned particle stays fixed, spring resists stretch (returns toward
 * rest), free particle falls under gravity, damping reduces oscillation,
 * stable over many steps, fail-closed on NULL/bad indices. */
#include "wubu_softbody.h"
#include <stdio.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)
#define NEAR(a,b,t) (fabs((a)-(b)) < (t))

int main(void) {
    wubu_softbody_t sb;
    CHECK(wubu_sb_init(&sb, 10.0) == 0, "init ok");

    /* a hanging mass: particle 0 pinned at top, particle 1 below, connected */
    int p0 = wubu_sb_add_particle(&sb, 0, 10, 1.0, 1);   /* pinned */
    int p1 = wubu_sb_add_particle(&sb, 0, 5, 1.0, 0);    /* free, 5 below */
    CHECK(p0 == 0 && p1 == 1, "particles added");
    int sp = wubu_sb_add_spring(&sb, p0, p1, 100.0, 5.0, 5.0);  /* rest 5 */
    CHECK(sp == 0, "spring added");

    /* pinned particle never moves */
    for (int i = 0; i < 100; i++) wubu_sb_step(&sb, 0.01);
    CHECK(NEAR(sb.p[0].x, 0.0, 1e-9) && NEAR(sb.p[0].y, 10.0, 1e-9), "pinned particle stays fixed");

    /* free particle: falls but spring holds it near rest (5 below pinned) */
    CHECK(sb.p[1].y < 10.0, "free particle below pinned");
    CHECK(sb.p[1].y > 0.0, "free particle didn't fall through (spring holds)");
    /* spring length near rest (small deformation) */
    double len = wubu_sb_spring_len(&sb, 0);
    CHECK(len > 3.0 && len < 7.0, "spring near rest length (deforms but holds)");

    /* stretch test: pull p1 down, spring pulls it back */
    wubu_softbody_t st;
    wubu_sb_init(&st, 0.0);   /* no gravity, isolate spring */
    wubu_sb_add_particle(&st, 0, 0, 1.0, 1);   /* pinned anchor */
    wubu_sb_add_particle(&st, 0, -10, 1.0, 0); /* free, stretched 10 below */
    wubu_sb_add_spring(&st, 0, 1, 50.0, 2.0, 5.0);  /* rest 5, stretched to 10 */
    double before = wubu_sb_spring_len(&st, 0);
    for (int i = 0; i < 50; i++) wubu_sb_step(&st, 0.01);
    double after = wubu_sb_spring_len(&st, 0);
    CHECK(after < before, "spring pulls stretched particle back toward rest");

    /* free particle falls under gravity (no spring) */
    wubu_softbody_t fg;
    wubu_sb_init(&fg, 10.0);
    wubu_sb_add_particle(&fg, 0, 100, 1.0, 0);
    for (int i = 0; i < 50; i++) wubu_sb_step(&fg, 0.01);
    CHECK(fg.p[0].y < 100.0 && fg.p[0].vy < 0, "free particle falls under gravity");

    /* damping reduces oscillation (KE decreases over time) */
    wubu_softbody_t dp;
    wubu_sb_init(&dp, 0.0);
    wubu_sb_add_particle(&dp, 0, 0, 1.0, 1);
    wubu_sb_add_particle(&dp, 0, -8, 1.0, 0);
    wubu_sb_add_spring(&dp, 0, 1, 80.0, 10.0, 5.0);   /* heavy damping */
    for (int i = 0; i < 20; i++) wubu_sb_step(&dp, 0.01);
    double ke1 = wubu_sb_ke(&dp);
    for (int i = 0; i < 100; i++) wubu_sb_step(&dp, 0.01);
    double ke2 = wubu_sb_ke(&dp);
    CHECK(ke2 < ke1, "damping reduces kinetic energy over time");

    /* stability: no NaN over many steps */
    int stable = 1;
    for (int i = 0; i < 200; i++) {
        wubu_sb_step(&sb, 0.005);
        if (isnan(sb.p[1].x) || isnan(sb.p[1].y)) { stable = 0; break; }
    }
    CHECK(stable == 1, "stable (no NaN over 200 steps)");

    /* fail-closed */
    CHECK(wubu_sb_step(NULL, 0.01) == -1, "NULL step -> -1");
    CHECK(wubu_sb_step(&sb, 0.0) == -1, "zero dt -> -1");
    CHECK(wubu_sb_add_particle(NULL, 0, 0, 1, 0) == -1, "NULL add_p -> -1");
    CHECK(wubu_sb_add_particle(&sb, 0, 0, 0, 0) == -1, "zero mass -> -1");
    CHECK(wubu_sb_add_spring(&sb, 0, 99, 1, 1, 1) == -1, "bad spring index -> -1");
    CHECK(wubu_sb_add_spring(&sb, 0, 0, 1, 1, 1) == -1, "self spring -> -1");
    CHECK(wubu_sb_spring_len(&sb, 99) == -1.0, "bad spring_len -> -1");

    printf("\n=== test_softbody: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
