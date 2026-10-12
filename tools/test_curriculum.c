/* test_curriculum.c — BRAIN #242: curriculum forward verified.
 *
 * Measured assertions on wubu_curriculum_threshold + eligible:
 *   1. threshold(0)==d0, threshold(1)==d1, threshold(0.5)==midpoint.
 *   2. monotonic non-decreasing in t (harder over time).
 *   3. t clamped to [0,1].
 *   4. eligibility: easy sample eligible early; hard sample only late.
 *   5. an easy sample is eligible at ALL t (never gated out).
 *   6. a too-hard sample is never eligible until the end.
 *   7. exact boundary: s == threshold -> eligible.
 */
#include "wubu_curriculum.h"
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
    printf("=== test_curriculum (curriculum learning, #242) ===\n");
    const double d0 = 0.2, d1 = 0.9;   /* easy -> hard difficulty */

    /* 1: endpoints + midpoint. */
    T(near(wubu_curriculum_threshold(d0, d1, 0.0), d0, 1e-9), "threshold(0)==d0");
    T(near(wubu_curriculum_threshold(d0, d1, 1.0), d1, 1e-9), "threshold(1)==d1");
    T(near(wubu_curriculum_threshold(d0, d1, 0.5), (d0+d1)/2, 1e-9), "threshold(0.5)==mid");

    /* 2: monotonic. */
    int mono = 1;
    double prev = -1e9;
    for (double t = 0.0; t <= 1.0001; t += 0.05) {
        double th = wubu_curriculum_threshold(d0, d1, t);
        if (th < prev - 1e-12) mono = 0;
        prev = th;
    }
    T(mono, "threshold monotonic non-decreasing in t");

    /* 3: clamp. */
    T(near(wubu_curriculum_threshold(d0, d1, -5.0), d0, 1e-9), "t<0 clamped to d0");
    T(near(wubu_curriculum_threshold(d0, d1, 5.0), d1, 1e-9), "t>1 clamped to d1");

    /* 4: easy early, hard late. */
    T(wubu_curriculum_eligible(d0, d1, 0.0, 0.2) == 1, "easy sample eligible at t=0");
    T(wubu_curriculum_eligible(d0, d1, 0.0, 0.9) == 0, "hard sample NOT eligible at t=0");
    T(wubu_curriculum_eligible(d0, d1, 1.0, 0.9) == 1, "hard sample eligible at t=1");

    /* 5: very easy sample always eligible. */
    int always = 1;
    for (double t = 0.0; t <= 1.0001; t += 0.1)
        if (!wubu_curriculum_eligible(d0, d1, t, 0.1)) always = 0;
    T(always, "very-easy sample eligible at all t");

    /* 6: too-hard never until end. */
    int never = 1;
    for (double t = 0.0; t < 1.0; t += 0.1)
        if (wubu_curriculum_eligible(d0, d1, t, 1.5)) never = 0;
    T(never, "impossibly-hard sample never eligible before t=1");

    /* 7: exact boundary eligible. */
    double th = wubu_curriculum_threshold(d0, d1, 0.3);
    T(wubu_curriculum_eligible(d0, d1, 0.3, th) == 1, "s == threshold -> eligible");

    printf("=== Results: %d/%d passed ===\n", g_pass, g_run);
    return g_pass == g_run ? 0 : 1;
}
