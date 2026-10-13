/* test_fixstep.c -- WORLDSIM #261 deterministic fixed-step simulation.
 * Proves: fixed dt steps decoupled from frame time, determinism (same total
 * elapsed -> same step count regardless of frame granularity), clamp prevents
 * spiral-of-death, alpha in [0,1), sim clock canonical, reset. */
#include "wubu_fixstep.h"
#include <stdio.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_fixstep_t f;
    double dt = 1.0/60.0;
    CHECK(wubu_fixstep_init(&f, dt, 5) == 0, "init ok");
    CHECK(wubu_fixstep_init(&f, 0.0, 5) == -1, "zero dt rejected");
    CHECK(wubu_fixstep_init(&f, -1.0, 5) == -1, "negative dt rejected");

    /* exactly one dt -> one step */
    CHECK(wubu_fixstep_advance(&f, dt) == 1, "one dt -> 1 step");
    /* half dt -> no step yet (carried) */
    CHECK(wubu_fixstep_advance(&f, dt/2) == 0, "half dt -> 0 steps");
    /* another half -> the carried + this = one step */
    CHECK(wubu_fixstep_advance(&f, dt/2) == 1, "carried + half -> 1 step");

    /* DETERMINISM: same total elapsed, different frame granularity -> same
     * total step count. Feed 1.0s as one big frame vs 60 small frames. */
    wubu_fixstep_t a, b;
    wubu_fixstep_init(&a, dt, 1000);
    wubu_fixstep_init(&b, dt, 1000);
    int sa = 0;
    sa += wubu_fixstep_advance(&a, 1.0);          /* one 1s frame */
    int sb = 0;
    for (int i = 0; i < 60; i++) sb += wubu_fixstep_advance(&b, 1.0/60.0);  /* 60 frames */
    CHECK(sa == 60, "one big frame -> 60 steps");
    CHECK(sb == 60, "60 small frames -> 60 steps");
    CHECK(sa == sb, "DETERMINISTIC: same steps regardless of frame granularity");

    /* clamp prevents spiral-of-death: a huge stall is capped */
    wubu_fixstep_t c;
    wubu_fixstep_init(&c, dt, 5);
    int capped = wubu_fixstep_advance(&c, 10.0);   /* 10s stall, clamp 5 */
    CHECK(capped == 5, "10s stall clamped to max_steps=5");
    CHECK(c.accumulator < dt, "excess dropped after clamp (no catch-up storm)");

    /* step() advances the canonical sim clock */
    wubu_fixstep_t d;
    wubu_fixstep_init(&d, dt, 5);
    int n = wubu_fixstep_advance(&d, dt*3);
    for (int i = 0; i < n; i++) wubu_fixstep_step(&d);
    CHECK(d.step_count == 3, "step_count == steps run");
    CHECK(fabs(d.sim_time - 3*dt) < 1e-12, "sim_time == step_count*dt (canonical clock)");

    /* alpha in [0,1) */
    double al = wubu_fixstep_alpha(&f);
    CHECK(al >= 0.0 && al < 1.0, "alpha in [0,1)");

    /* reset */
    wubu_fixstep_reset(&d);
    CHECK(d.step_count == 0 && d.sim_time == 0.0 && d.accumulator == 0.0, "reset clears clock+accumulator");

    /* fail-closed on NULL */
    CHECK(wubu_fixstep_advance(NULL, 1.0) == 0, "NULL advance -> 0");
    CHECK(wubu_fixstep_alpha(NULL) == 0.0, "NULL alpha -> 0");

    printf("\n=== test_fixstep: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
