/* test_qk_norm.c — BRAIN #235: query-key norm forward verified.
 *
 * Measured assertions on wubu_qk_rmsnorm:
 *   1. output RMS == gamma (the defining property: RMSNorm bounds RMS to gamma).
 *   2. sign preserved: output[i] has same sign as input[i].
 *   3. constant vector -> all entries == gamma (x/sqrt(x^2)=sign(x)).
 *   4. zero vector stays zero (eps prevents div-by-zero, no NaN).
 *   5. scaling invariance: RMSNorm(c*x) == RMSNorm(x) for c>0.
 *   6. bad args rejected.
 */
#include "wubu_query_key_norm.h"
#include <stdio.h>
#include <math.h>

static int g_run = 0, g_pass = 0;
#define T(cond, msg) do { \
    g_run++; \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while (0)

static int near(double a, double b, double eps) { return fabs(a - b) <= eps; }

static double rms(const float *x, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    return sqrt(ss / n);
}

int main(void) {
    printf("=== test_qk_norm (query-key norm, #235) ===\n");

    const float gamma = 1.5f, eps = 1e-6f;

    /* 1: output RMS == gamma. */
    float x[8] = {1,-2,3,-4,5,-6,7,-8};
    float orig[8]; for (int i=0;i<8;i++) orig[i]=x[i];
    T(wubu_qk_rmsnorm(x, 8, gamma, eps) == 0, "rmsnorm ok");
    T(near(rms(x, 8), gamma, 1e-4), "output RMS == gamma");

    /* 2: sign preserved. */
    int sign_ok = 1;
    for (int i = 0; i < 8; i++) {
        if ((orig[i] > 0 && x[i] <= 0) || (orig[i] < 0 && x[i] >= 0)) sign_ok = 0;
    }
    T(sign_ok, "sign preserved");

    /* 3: constant vector -> all == gamma. */
    float c[4] = {3,3,3,3};
    wubu_qk_rmsnorm(c, 4, gamma, eps);
    int const_ok = 1;
    for (int i = 0; i < 4; i++) if (!near(c[i], gamma, 1e-4)) const_ok = 0;
    T(const_ok, "constant vector -> all == gamma");

    /* 4: zero vector -> zero, no NaN. */
    float z[4] = {0,0,0,0};
    wubu_qk_rmsnorm(z, 4, gamma, eps);
    int z_ok = 1;
    for (int i = 0; i < 4; i++) if (z[i] != 0.0f || isnan(z[i])) z_ok = 0;
    T(z_ok, "zero vector stays zero (no NaN)");

    /* 5: scaling invariance RMSNorm(c*x)==RMSNorm(x). */
    float a[6] = {1,-2,3,-4,5,-6};
    float b[6]; for (int i=0;i<6;i++) b[i]=a[i]*7.0f;
    wubu_qk_rmsnorm(a, 6, gamma, eps);
    wubu_qk_rmsnorm(b, 6, gamma, eps);
    int inv = 1;
    for (int i = 0; i < 6; i++) if (!near(a[i], b[i], 1e-4)) inv = 0;
    T(inv, "scaling invariance: RMSNorm(c*x)==RMSNorm(x)");

    /* 6: bad args. */
    T(wubu_qk_rmsnorm(x, 0, gamma, eps) == -1, "n<=0 rejected");
    T(wubu_qk_rmsnorm(NULL, 8, gamma, eps) == -1, "NULL rejected");
    T(wubu_qk_rmsnorm(x, 8, gamma, 0.0f) == -1, "eps<=0 rejected");

    printf("=== Results: %d/%d passed ===\n", g_pass, g_run);
    return g_pass == g_run ? 0 : 1;
}
