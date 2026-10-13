/* test_worldsim.c -- WORLDSIM #257 worldsim solid dynamic system.
 * Proves: bodies fall under gravity, collision resolves, world is
 * deterministic (same elapsed -> same state), energy behavior, fail-closed. */
#include "wubu_worldsim.h"
#include <stdio.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)
#define NEAR(a,b,t) (fabs((a)-(b)) < (t))

int main(void) {
    wubu_pforces_t grav = {10.0, 0.0, 0.0};   /* gravity 10, no drag */
    wubu_world_t w;
    CHECK(wubu_world_init(&w, 2, grav, 0.5, 1.0/60.0, 5) == 0, "world init ok");

    /* add a falling body high up */
    int bi = wubu_world_add(&w, 0, 100, 0, 0, 1, 1);
    CHECK(bi == 0, "body added at index 0");
    const wubu_body_t *b0 = wubu_world_body(&w, 0);
    CHECK(NEAR(b0->y, 100.0, 1e-9), "body starts at y=100");

    /* step 0.5s: body should have fallen */
    int steps = wubu_world_step(&w, 0.5);
    CHECK(steps > 0, "world advanced (steps > 0)");
    b0 = wubu_world_body(&w, 0);
    CHECK(b0->y < 100.0, "body fell under gravity");
    CHECK(b0->vy < 0.0, "body gained downward velocity");

    /* DETERMINISM: two identical worlds, same stepping -> identical state */
    wubu_world_t w1, w2;
    wubu_world_init(&w1, 1, grav, 0.5, 1.0/60.0, 1000);
    wubu_world_init(&w2, 1, grav, 0.5, 1.0/60.0, 1000);
    wubu_world_add(&w1, 0, 50, 3, 0, 1, 1);
    wubu_world_add(&w2, 0, 50, 3, 0, 1, 1);
    wubu_world_step(&w1, 1.0);
    wubu_world_step(&w2, 1.0);
    const wubu_body_t *a = wubu_world_body(&w1, 0);
    const wubu_body_t *c = wubu_world_body(&w2, 0);
    CHECK(NEAR(a->x, c->x, 1e-12) && NEAR(a->y, c->y, 1e-12), "DETERMINISTIC: identical worlds -> identical state");

    /* collision: two bodies moving toward each other bounce */
    wubu_world_t wc;
    wubu_world_init(&wc, 2, grav, 1.0, 1.0/60.0, 1000);   /* elastic */
    wubu_world_add(&wc, 0, 0, 5, 0, 1, 1);      /* moving +x */
    wubu_world_add(&wc, 3, 0, 0, 0, 1, 1);      /* at rest, will be hit */
    for (int i = 0; i < 120; i++) wubu_world_step(&wc, 1.0/60.0);  /* 2s */
    const wubu_body_t *ca = wubu_world_body(&wc, 0);
    const wubu_body_t *cb = wubu_world_body(&wc, 1);
    CHECK(cb->vx > 0, "collision transferred momentum to second body");
    CHECK(ca->vx < 5.0, "first body slowed after collision");

    /* KE accounting works */
    CHECK(wubu_world_ke(&wc) > 0.0, "world KE > 0 after motion");

    /* fail-closed */
    CHECK(wubu_world_step(NULL, 1.0) == 0, "NULL world -> 0 steps");
    CHECK(wubu_world_add(&w, 0, 0, 0, 0, 0, 1) == -1, "zero mass body rejected");
    CHECK(wubu_world_body(&w, 999) == NULL, "out-of-range body -> NULL");
    CHECK(wubu_world_body(NULL, 0) == NULL, "NULL world body -> NULL");

    printf("\n=== test_worldsim: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
