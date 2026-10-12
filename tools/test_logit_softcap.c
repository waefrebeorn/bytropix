/* test_logit_softcap.c — BRAIN #239: logit softcap forward verified.
 *
 * Measured assertions on wubu_softcap + wubu_softcap_arr:
 *   1. bounded: |softcap(x)| < cap for all x (including huge x).
 *   2. softcap(0) == 0 (odd function, fixes origin).
 *   3. odd: softcap(-x) == -softcap(x).
 *   4. ~linear near 0: softcap(0.001*cap) ~ 0.001*cap (within 1e-4 rel).
 *   5. saturates: softcap(1000*cap) ~ cap (approaches the bound).
 *   6. exact: softcap at known point matches cap*tanh(x/cap).
 *   7. cap<=0 -> identity; bad args rejected.
 */
#include "wubu_logit_softcap.h"
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
    printf("=== test_logit_softcap (logit softcap, #239) ===\n");
    const double cap = 30.0;   /* Gemma-2 default */

    /* 1: bounded. */
    T(fabs(wubu_softcap(1e6, cap)) <= cap, "huge x bounded (<= cap)");
    T(fabs(wubu_softcap(-1e6, cap)) <= cap, "huge -x bounded");

    /* 2: origin. */
    T(near(wubu_softcap(0.0, cap), 0.0, 1e-12), "softcap(0)==0");

    /* 3: odd. */
    T(near(wubu_softcap(-12.3, cap), -wubu_softcap(12.3, cap), 1e-9), "odd function");

    /* 4: ~linear near 0. */
    double small = 0.001 * cap;
    T(near(wubu_softcap(small, cap), small, 1e-3), "~linear near 0");

    /* 5: saturates. */
    T(near(wubu_softcap(1000.0*cap, cap), cap, 1e-6), "saturates to cap");

    /* 6: exact value. */
    T(near(wubu_softcap(15.0, cap), cap*tanh(15.0/cap), 1e-9), "exact cap*tanh(x/cap)");

    /* 7: identity on cap<=0 + arr + bad args. */
    T(near(wubu_softcap(5.0, 0.0), 5.0, 1e-12), "cap<=0 identity");
    float l[4] = {1e5f, -1e5f, 0.0f, 10.0f};
    T(wubu_softcap_arr(l, 4, cap) == 0, "arr ok");
    int bounded = 1;
    for (int i = 0; i < 4; i++) if (fabs(l[i]) > cap + 1e-3f) bounded = 0;
    T(bounded, "arr output all bounded by cap");
    T(wubu_softcap_arr(NULL, 4, cap) == -1, "NULL rejected");
    T(wubu_softcap_arr(l, 0, cap) == -1, "n<=0 rejected");

    printf("=== Results: %d/%d passed ===\n", g_pass, g_run);
    return g_pass == g_run ? 0 : 1;
}
