/* test_physics.c -- WORLDSIM #264 gravity + drag + restitution tuned forces.
 * Proves: free-fall accelerates at g, drag reaches terminal velocity, energy
 * conserved without drag, drag dissipates energy, tunability (feather vs
 * free-fall), terminal-velocity formula, fail-closed on NULL. */
#include "wubu_physics.h"
#include <stdio.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)
#define NEAR(a,b,t) (fabs((a)-(b)) < (t))

int main(void) {
    wubu_pforces_t f;

    /* free fall (no drag): after 1s at g=10, v ~ -10 */
    wubu_physics_init(&f, 10.0, 0.0, 0.0);
    wubu_pbody_t b = {0, 100, 0, 0};
    double dt = 0.01;
    for (int i = 0; i < 100; i++) wubu_physics_step(&b, &f, dt);   /* 1s */
    CHECK(NEAR(b.vy, -10.0, 0.15), "free fall: v ~ -g*t after 1s");
    CHECK(b.y < 100, "free fall: body descends");

    /* energy conserved without drag (KE + PE constant) */
    wubu_pbody_t e1 = {0, 100, 5, 0};
    double e0 = wubu_physics_energy(&e1, &f);
    for (int i = 0; i < 50; i++) wubu_physics_step(&e1, &f, dt);
    double e1b = wubu_physics_energy(&e1, &f);
    CHECK(NEAR(e0, e1b, 0.5), "no drag: energy ~ conserved");

    /* with drag: reaches terminal velocity, energy dissipates */
    wubu_physics_init(&f, 10.0, 0.0, 0.5);   /* quad drag */
    double vterm = wubu_physics_terminal_velocity(&f);
    CHECK(vterm > 0, "terminal velocity > 0 with drag");
    wubu_pbody_t d = {0, 1000, 0, 0};
    for (int i = 0; i < 2000; i++) wubu_physics_step(&d, &f, dt);  /* long fall */
    CHECK(NEAR(-d.vy, vterm, vterm*0.05), "reaches terminal velocity under drag");

    /* tunability: high drag = slow (feather), no drag = fast (free fall) */
    wubu_physics_init(&f, 10.0, 5.0, 0.0);   /* high linear drag */
    double vt_feather = wubu_physics_terminal_velocity(&f);
    CHECK(vt_feather < vterm, "high drag -> lower terminal velocity (feather)");

    /* linear-drag terminal velocity formula: v = g/k */
    CHECK(NEAR(vt_feather, 10.0/5.0, 1e-9), "linear drag terminal v = g/k");

    /* no drag -> terminal velocity 0 (free fall, unbounded) */
    wubu_physics_init(&f, 10.0, 0.0, 0.0);
    CHECK(wubu_physics_terminal_velocity(&f) == 0.0, "no drag -> terminal v 0");

    /* drag dissipates energy (body falls, KE plateaus, PE drops) */
    wubu_physics_init(&f, 10.0, 0.0, 0.5);
    wubu_pbody_t g = {0, 1000, 0, 0};
    double eg0 = wubu_physics_energy(&g, &f);
    for (int i = 0; i < 500; i++) wubu_physics_step(&g, &f, dt);
    double eg1 = wubu_physics_energy(&g, &f);
    CHECK(eg1 < eg0, "drag dissipates energy");

    /* fail-closed on NULL */
    CHECK(wubu_physics_step(NULL, &f, dt) == -1, "NULL body -> -1");
    CHECK(wubu_physics_step(&g, NULL, dt) == -1, "NULL forces -> -1");
    CHECK(wubu_physics_step(&g, &f, 0.0) == -1, "zero dt -> -1");
    CHECK(wubu_physics_init(NULL, 1, 1, 1) == -1, "NULL init -> -1");

    printf("\n=== test_physics: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
