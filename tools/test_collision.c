/* test_collision.c -- WORLDSIM #258 rigid-body collision response (solid).
 * Proves: overlap detection, head-on elastic bounce (momentum conserved,
 * velocity reversed for equal masses), restitution damping (inelastic loses
 * KE), unequal-mass transfer, no-collision when apart/separating, positional
 * de-overlap, fail-closed on NULL/zero-mass. */
#include "wubu_collision.h"
#include <stdio.h>
#include <math.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)
#define NEAR(a,b,t) (fabs((a)-(b)) < (t))

int main(void) {
    /* two equal-mass bodies colliding head-on (a moving +x, b at rest) */
    wubu_body_t a = {0,0, 5,0, 1,1};
    wubu_body_t b = {1.5,0, 0,0, 1,1};   /* touching-ish (dist 1.5 < rsum 2) */
    CHECK(wubu_bodies_overlap(&a,&b) == 1, "overlap detected");

    /* not overlapping when apart */
    wubu_body_t c = {0,0, 0,0, 1,1};
    wubu_body_t d = {5,0, 0,0, 1,1};
    CHECK(wubu_bodies_overlap(&c,&d) == 0, "apart bodies not overlapping");

    /* elastic head-on (e=1), equal mass: a stops, b takes a's velocity */
    wubu_body_t p = {0,0, 5,0, 1,1};
    wubu_body_t q = {1.5,0, 0,0, 1,1};
    double ke_before = wubu_body_ke(&p) + wubu_body_ke(&q);
    int hit = wubu_collision_resolve(&p, &q, 1.0);
    CHECK(hit == 1, "collision resolved");
    CHECK(NEAR(p.vx, 0.0, 1e-9), "elastic equal-mass: a stops");
    CHECK(NEAR(q.vx, 5.0, 1e-9), "elastic equal-mass: b takes a's velocity");
    double ke_after = wubu_body_ke(&p) + wubu_body_ke(&q);
    CHECK(NEAR(ke_before, ke_after, 1e-9), "elastic collision conserves KE");
    /* momentum conserved */
    double mom_before = 1*5 + 1*0, mom_after = p.mass*p.vx + q.mass*q.vx;
    CHECK(NEAR(mom_before, mom_after, 1e-9), "momentum conserved");

    /* inelastic (e=0): they move together, KE lost */
    wubu_body_t r = {0,0, 5,0, 1,1};
    wubu_body_t s = {1.5,0, 0,0, 1,1};
    double ke0 = wubu_body_ke(&r) + wubu_body_ke(&s);
    wubu_collision_resolve(&r, &s, 0.0);
    CHECK(NEAR(r.vx, s.vx, 1e-9), "inelastic: equal velocity after (move together)");
    double ke1 = wubu_body_ke(&r) + wubu_body_ke(&s);
    CHECK(ke1 < ke0, "inelastic collision loses KE");

    /* unequal mass: light body bounces back off heavy stationary body.
     * heavy at x=0, light at x=-1.5 moving +x TOWARD heavy (approaching). */
    wubu_body_t heavy = {0,0, 0,0, 100,1};
    wubu_body_t light = {-1.5,0, 3,0, 1,1};   /* light moving +x into heavy */
    wubu_collision_resolve(&light, &heavy, 1.0);
    CHECK(light.vx < 0, "light body bounces back off heavy (elastic)");
    CHECK(heavy.vx < 0.1, "heavy body barely moves (v << light's)");

    /* separating bodies: no impulse (b moving away from a along +normal) */
    wubu_body_t x1 = {0,0, 5,0, 1,1};
    wubu_body_t x2 = {1.5,0, 5,0, 1,1};   /* both moving +x at same speed = not approaching */
    int nohit = wubu_collision_resolve(&x1, &x2, 1.0);
    CHECK(nohit == 0, "co-moving bodies: no collision impulse");

    /* positional de-overlap: after resolve, separated to within the skin
     * tolerance (overlap reduced to <= skin, so no visible sinking). */
    wubu_body_t y1 = {0,0, 5,0, 1,1};
    wubu_body_t y2 = {1.0,0, 0,0, 1,1};    /* deep overlap (dist 1.0 < 2.0) */
    double dist_before = 1.0;
    wubu_collision_resolve(&y1, &y2, 0.5);
    double dx = y2.x - y1.x, dy = y2.y - y1.y;
    double dist_after = sqrt(dx*dx + dy*dy);
    CHECK(dist_after > dist_before + 0.5, "positional correction pushes bodies apart");
    CHECK(dist_after >= 2.0 - 0.02, "separated to within skin tolerance");

    /* fail-closed */
    CHECK(wubu_collision_resolve(NULL, &q, 1.0) == 0, "NULL a -> no resolve");
    wubu_body_t zm = {0,0,0,0, 0,1};       /* zero mass */
    CHECK(wubu_collision_resolve(&zm, &q, 1.0) == 0, "zero mass -> no resolve (fail-closed)");

    printf("\n=== test_collision: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
