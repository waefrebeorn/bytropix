/* test_zloss.c — BRAIN #237: z-loss forward verified.
 *
 * Measured assertions on wubu_logsumexp + wubu_zloss:
 *   1. logsumexp(all zeros, n) == log(n).
 *   2. logsumexp shift-invariant: lse(x+c) == lse(x)+c (numerical stability).
 *   3. large logits (1e4) don't overflow (max-shift works).
 *   4. z-loss == z_coef * lse^2 (exact).
 *   5. z-loss >= 0 and grows with logit magnitude (penalizes blow-up).
 *   6. z_coef=0 -> 0 penalty.
 */
#include "wubu_zloss.h"
#include <stdio.h>
#include <math.h>

static int g_run = 0, g_pass = 0;
#define T(cond, msg) do { \
    g_run++; \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while (0)

static int near(double a, double b, double eps) { return fabs(a - b) <= eps; }

int main(void) {
    printf("=== test_zloss (z-loss regularizer, #237) ===\n");

    /* 1: lse(zeros, n) == log(n). */
    float z[4] = {0,0,0,0};
    T(near(wubu_logsumexp(z, 4), log(4.0), 1e-6), "lse(zeros,4)==log(4)");

    /* 2: shift invariance. */
    float x[3] = {1,2,3};
    float xc[3] = {1+100, 2+100, 3+100};
    T(near(wubu_logsumexp(xc, 3), wubu_logsumexp(x, 3) + 100.0, 1e-3),
      "lse shift-invariant (lse(x+c)==lse(x)+c)");

    /* 3: no overflow on huge logits. */
    float big[3] = {10000, 10001, 10002};
    double lb = wubu_logsumexp(big, 3);
    T(isfinite(lb) && lb > 10000, "large logits: no overflow");

    /* 4: z-loss == z_coef*lse^2. */
    float l[3] = {1,2,3};
    double lse = wubu_logsumexp(l, 3);
    T(near(wubu_zloss(l, 3, 0.001), 0.001 * lse * lse, 1e-9), "z-loss == coef*lse^2");

    /* 5: grows with magnitude; >=0. */
    float small[3] = {0.1f, 0.2f, 0.3f};
    double zs = wubu_zloss(small, 3, 0.001);
    double zl = wubu_zloss(l, 3, 0.001);
    T(zs >= 0.0 && zl > zs, "z-loss >=0, grows with logit magnitude");

    /* 6: coef 0 -> 0. */
    T(wubu_zloss(l, 3, 0.0) == 0.0, "z_coef=0 -> 0 penalty");

    printf("=== Results: %d/%d passed ===\n", g_pass, g_run);
    return g_pass == g_run ? 0 : 1;
}
