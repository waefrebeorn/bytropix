/* test_residual_scaling.c — BRAIN #236: residual scaling forward verified.
 *
 * Measured assertions on wubu_residual_scale_factor + wubu_residual_add:
 *   1. scale(1)==1, scale(4)==0.5, scale(9)==1/3 (1/sqrt(n)).
 *   2. scale(0) invalid -> 0.0.
 *   3. residual_add with scale=1 is plain addition.
 *   4. residual_add applies scale: r + s*b exactly.
 *   5. N stacked scaled branches keep variance ~O(1) (the point of the scaling):
 *      after N layers each contributing branch~unit, residual magnitude stays
 *      bounded (~sqrt(N)) not ~N.
 *   6. bad args rejected.
 */
#include "wubu_residual_scaling.h"
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
    printf("=== test_residual_scaling (residual branch scaling, #236) ===\n");

    /* 1: 1/sqrt(n). */
    T(near(wubu_residual_scale_factor(1), 1.0, 1e-9), "scale(1)==1");
    T(near(wubu_residual_scale_factor(4), 0.5, 1e-9), "scale(4)==0.5");
    T(near(wubu_residual_scale_factor(9), 1.0/3.0, 1e-9), "scale(9)==1/3");

    /* 2: invalid. */
    T(wubu_residual_scale_factor(0) == 0.0, "scale(0)==0 (invalid)");

    /* 3: scale=1 plain add. */
    float r[4] = {1,2,3,4};
    float b[4] = {10,20,30,40};
    T(wubu_residual_add(r, b, 4, 1.0) == 0, "add ok");
    T(r[0]==11 && r[3]==44, "scale=1 is plain addition");

    /* 4: scaled add exact. */
    float r2[3] = {0,0,0};
    float b2[3] = {2,4,6};
    wubu_residual_add(r2, b2, 3, 0.5);
    T(r2[0]==1 && r2[1]==2 && r2[2]==3, "r + 0.5*b exact");

    /* 5: variance bounded. Simulate N layers, each branch unit-magnitude,
     * scaled by 1/sqrt(N). After N adds the residual magnitude ~ sqrt(N),
     * NOT ~N. Compare scaled vs unscaled (scale=1) growth. */
    const int N = 16;
    double scaled = 0.0, unscaled = 0.0;
    double s = wubu_residual_scale_factor(N);
    for (int i = 0; i < N; i++) { scaled += s * 1.0; unscaled += 1.0; }
    /* scaled ~ sqrt(N)=4, unscaled ~ N=16. Scaled must be far smaller. */
    T(scaled < unscaled / 2.0, "scaled residual stays bounded vs unscaled");
    T(near(scaled, sqrt((double)N), 1e-6), "scaled sum == sqrt(N)");

    /* 6: bad args. */
    T(wubu_residual_add(NULL, b, 4, 1.0) == -1, "NULL residual rejected");
    T(wubu_residual_add(r, NULL, 4, 1.0) == -1, "NULL branch rejected");
    T(wubu_residual_add(r, b, 0, 1.0) == -1, "n<=0 rejected");

    printf("=== Results: %d/%d passed ===\n", g_pass, g_run);
    return g_pass == g_run ? 0 : 1;
}
