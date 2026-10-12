/* test_alibi.c — BRAIN #232: ALiBi forward verified.
 *
 * Measured assertions on wubu_alibi_slopes + wubu_alibi_bias:
 *   1. slopes in (0,1), strictly decreasing, head-0 == 2^(-8/n).
 *   2. bias(i,i) == 0 (no self-penalty).
 *   3. bias is symmetric: bias(i,j) == bias(j,i).
 *   4. bias negative and grows with distance: bias(i,i+d) == -m*d.
 *   5. exact value check: slope 0.5, dist 3 -> -1.5.
 *   6. bad args rejected.
 */
#include "wubu_alibi.h"
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
    printf("=== test_alibi (ALiBi linear attention bias, #232) ===\n");

    const int n = 8;
    double s[8];
    T(wubu_alibi_slopes(n, s) == 0, "slopes computed");

    /* 1: in (0,1), strictly decreasing, head-0 exact. */
    int ok = 1;
    for (int i = 0; i < n; i++) if (!(s[i] > 0.0 && s[i] < 1.0)) ok = 0;
    for (int i = 1; i < n; i++) if (!(s[i] < s[i-1])) ok = 0;
    T(ok, "slopes in (0,1), strictly decreasing");
    T(near(s[0], pow(2.0, -8.0/n), 1e-12), "head-0 slope == 2^(-8/n)");

    /* 2: no self-penalty. */
    T(near(wubu_alibi_bias(s[0], 5, 5), 0.0, 1e-12), "bias(i,i) == 0");

    /* 3: symmetric. */
    T(near(wubu_alibi_bias(s[2], 3, 7), wubu_alibi_bias(s[2], 7, 3), 1e-12),
      "bias symmetric in (i,j)");

    /* 4: negative, linear in distance. */
    double b1 = wubu_alibi_bias(s[0], 0, 1);
    double b3 = wubu_alibi_bias(s[0], 0, 3);
    T(b1 < 0.0 && b3 < b1, "bias negative, grows with distance");
    T(near(b3, 3.0 * b1, 1e-12), "bias linear in distance (b3 == 3*b1)");

    /* 5: exact value: slope 0.5, dist 3 -> -1.5. */
    T(near(wubu_alibi_bias(0.5, 0, 3), -1.5, 1e-12), "exact bias(0.5, dist 3) == -1.5");

    /* 6: bad args. */
    T(wubu_alibi_slopes(0, s) == -1, "n_heads<=0 rejected");
    T(wubu_alibi_slopes(n, NULL) == -1, "NULL slopes rejected");

    printf("=== Results: %d/%d passed ===\n", g_pass, g_run);
    return g_pass == g_run ? 0 : 1;
}
