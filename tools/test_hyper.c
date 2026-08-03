/*
 * test_hyper.c -- verify the LEAN-PROVEN hyperbolic properties in C11.
 *
 * Each test corresponds to a formal theorem in MATH/lean/wubu_proofs/:
 *   MobiusAdd.lean:          mobius_add preserves the ball
 *   PoincareBall.lean:       exp_0^c(log_0^c(y)) = y
 *   HyperbolicGyration.lean: gyroassociativity (ball points)
 * The math is PROVEN; the test pins the implementation to the proof.
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "wubu_hyper.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

int main(void)
{
    printf("=== test_hyper (the Lean-verified hyperbolic layer) ===\n");
    double c = 0.25;   /* curvature */

    /* 1. MobiusAdd: x⊕y stays in the ball when x,y are (|.|² < 1/c) */
    {
        srand(7);
        int ok = 1, cnt = 0;
        for (int t = 0; t < 20000; t++) {
            double x = ((double)rand() / RAND_MAX) * 1.6 - 0.8;   /* |x| < 0.8 < 2=1/√c... use 1/c=4, so |x|<2 */
            double y = ((double)rand() / RAND_MAX) * 1.6 - 0.8;
            double z = wubu_hyper_mobius_add(c, x, y);
            cnt++;
            if (!(z * z < 1.0 / c + 1e-9)) { ok = 0; break; }
        }
        CHECK(ok, "mobius addition preserves the ball (Lean: mobius_add_preserves_ball)");
        printf("  mobius closure: %d samples in ball\n", cnt);
    }

    /* 2. PoincareBall: exp(log(y)) == y for ball points */
    {
        double y = 0.5;
        double v = wubu_hyper_log(c, y);
        double y2 = wubu_hyper_exp(c, v);
        CHECK(fabs(y2 - y) < 1e-12,
              "exp(log(y)) == y (Lean: poincare_exp_log_identity_nonzero)");
        printf("  exp(log(%.3f)) = %.15f\n", y, y2);
    }

    /* 3. vector exp/log round trip */
    {
        double v[8] = {0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8};
        double y[8], back[8];
        wubu_hyper_exp_vec(c, v, y, 8);
        wubu_hyper_log_vec(c, y, back, 8);
        double err = 0;
        for (int i = 0; i < 8; i++) err += fabs(back[i] - v[i]);
        CHECK(err < 1e-9, "vector exp/log round trip");
        /* the lifted point must be inside the ball: ‖y‖ < 1/√c */
        double ny = 0; for (int i = 0; i < 8; i++) ny += y[i]*y[i];
        ny = sqrt(ny);
        CHECK(ny < 1.0 / sqrt(c), "lifted point inside the ball");
        printf("  vector round trip err %.2e, ball norm %.4f (limit %.2f)\n",
               err, ny, 1.0 / sqrt(c));
    }

    /* 4. gyroassociativity on ball points (Lean: gyroassoc_1d) */
    {
        double u = 0.3, v = -0.4, w = 0.5;
        double uv = wubu_hyper_mobius_add(c, u, v);
        double vw = wubu_hyper_mobius_add(c, v, w);
        double lhs = wubu_hyper_mobius_add(c, u, vw);
        double rhs = wubu_hyper_mobius_add(c, uv, w);
        CHECK(fabs(lhs - rhs) < 1e-12,
              "gyroassociativity (Lean: gyroassoc_1d)");
        printf("  gyroassoc: u⊕(v⊕w)=%.15f (u⊕v)⊕w=%.15f\n", lhs, rhs);
    }

    /* 5. the distance is positive and symmetric */
    {
        double x[4] = {0.1, 0.2, 0.3, 0.4};
        double y[4] = {-0.1, 0.0, 0.5, -0.2};
        double d1 = wubu_hyper_dist(c, x, y, 4);
        double d2 = wubu_hyper_dist(c, y, x, 4);
        CHECK(d1 > 0 && fabs(d1 - d2) < 1e-12, "distance positive + symmetric");
        printf("  distance d(x,y)=%.6f d(y,x)=%.6f\n", d1, d2);
    }

    /* 6. the model-facing layer: lift -> gyro-rotate -> project */
    {
        wubu_hyper_t h;
        h.c = 0.5;
        h.n = 8;
        double x[8] = {0.2, -0.1, 0.3, 0.05, -0.2, 0.1, -0.3, 0.15};
        double y[8], k[8], out[8], back[8];
        CHECK(wubu_hyper_lift(&h, x, y) == 0, "lift");
        /* gyrovector identity: (-y) ⊕ y = 0 */
        wubu_hyper_gyro_rotate(&h, y, y, out);
        double zero_norm = 0; for (int i = 0; i < 8; i++) zero_norm += out[i]*out[i];
        CHECK(sqrt(zero_norm) < 1e-12, "gyro rotate: (-y)⊕y = 0 (gyrovector identity)");
        /* rotate the query by a DIFFERENT key: q' = (-k)⊕q stays in the ball */
        double kx[8] = {-0.3, 0.2, -0.1, 0.4, -0.05, 0.3, -0.2, 0.25};
        wubu_hyper_lift(&h, kx, k);
        wubu_hyper_gyro_rotate(&h, y, k, out);
        double on = 0; for (int i = 0; i < 8; i++) on += out[i]*out[i];
        on = sqrt(on);
        CHECK(on < 1.0 / sqrt(h.c) + 1e-9, "rotated query inside the ball");
        /* project back: the rotated query returns to a valid tangent point */
        CHECK(wubu_hyper_project(&h, out, back) == 0, "project");
        printf("  layer: (-y)⊕y norm %.2e, rotated-query ball norm %.4f\n",
               sqrt(zero_norm), on);
    }

    if (failures == 0) printf("ALL HYPER TESTS PASSED -- the proven math runs\n");
    else printf("%d HYPER FAILURES\n", failures);
    return failures ? 1 : 0;
}
