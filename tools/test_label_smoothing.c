/* test_label_smoothing.c — BRAIN #241: label smoothing forward verified.
 *
 * Measured assertions on wubu_label_smooth + wubu_label_smooth_loss:
 *   1. alpha=0 -> one-hot (target[y]=1, rest 0).
 *   2. sum(target) == 1 (valid distribution) for any alpha.
 *   3. target[y] == 1 - alpha + alpha/V; others == alpha/V (exact).
 *   4. smoothing moves mass off the true label (target[y] < 1 for alpha>0).
 *   5. uniform alpha=1 -> all classes equal 1/V.
 *   6. loss: perfect confident logp on y -> ~ (1-target[y])*(-log(1-eps)) small;
 *       loss against uniform target == -mean(logp).
 *   7. bad args rejected.
 */
#include "wubu_label_smoothing.h"
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
    printf("=== test_label_smoothing (label smoothing, #241) ===\n");
    const int V = 5, y = 2;
    const double alpha = 0.1;
    float t[5];

    /* 1: alpha=0 one-hot. */
    T(wubu_label_smooth(t, V, y, 0.0) == 0, "smooth ok");
    T(t[y]==1.0f && t[0]==0.0f && t[4]==0.0f, "alpha=0 -> one-hot");

    /* 2: sums to 1. */
    wubu_label_smooth(t, V, y, alpha);
    double sum = 0; for (int i=0;i<V;i++) sum += t[i];
    T(near(sum, 1.0, 1e-6), "sum(target)==1");

    /* 3: exact values. */
    double off = alpha / V;
    T(near(t[y], 1.0 - alpha + off, 1e-6), "target[y]==1-alpha+alpha/V");
    int off_ok = 1;
    for (int i=0;i<V;i++) if (i!=y && !near(t[i], off, 1e-6)) off_ok = 0;
    T(off_ok, "others == alpha/V");

    /* 4: mass moved off true label. */
    T(t[y] < 1.0f, "alpha>0 reduces target[y] below 1");

    /* 5: alpha=1 uniform. */
    wubu_label_smooth(t, V, y, 1.0);
    int uni = 1; for (int i=0;i<V;i++) if (!near(t[i], 1.0/V, 1e-6)) uni = 0;
    T(uni, "alpha=1 -> uniform 1/V");

    /* 6: loss against uniform target == -mean(logp). */
    float logp[5] = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f};
    double mean_lp = (-1.0-2.0-3.0-4.0-5.0)/5.0;
    T(near(wubu_label_smooth_loss(t, logp, V), -mean_lp, 1e-6), "uniform loss == -mean(logp)");
    /* loss against one-hot y == -logp[y]. */
    wubu_label_smooth(t, V, y, 0.0);
    T(near(wubu_label_smooth_loss(t, logp, V), -logp[y], 1e-6), "one-hot loss == -logp[y]");

    /* 7: bad args. */
    T(wubu_label_smooth(NULL, V, y, alpha) == -1, "NULL rejected");
    T(wubu_label_smooth(t, V, V, alpha) == -1, "y out of range rejected");
    T(wubu_label_smooth(t, V, y, 1.5) == -1, "alpha>1 rejected");
    T(wubu_label_smooth_loss(t, NULL, V) < 0, "NULL logp rejected");

    printf("=== Results: %d/%d passed ===\n", g_pass, g_run);
    return g_pass == g_run ? 0 : 1;
}
