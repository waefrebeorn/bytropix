/* wubu_passk.c -- the pass@k estimator. */
#include <math.h>
#include "wubu_passk.h"

/* log C(n, k) via the log-gamma -- the counts are huge, the ratio is not */
static double lchoose(int n, int k)
{
    if (k < 0 || k > n || n < 0) return -1e300;
    double r = 0;
    for (int i = 1; i <= k; i++) r += log((double)(n - k + i) / (double)i);
    return r;
}

float wubu_passk(const int *succ, int n, int k)
{
    if (!succ || n < 1 || k < 1 || k > n) return 0;
    int s = 0;
    for (int i = 0; i < n; i++) if (succ[i]) s++;
    if (s == 0) return 0;
    if (s >= n || k >= n - s + 1) return 1;
    /* pass@k = 1 - C(n-s, k)/C(n, k) */
    double d = lchoose(n - s, k) - lchoose(n, k);
    double p = 1.0 - exp(d);
    return (float)(p < 0 ? 0 : (p > 1 ? 1 : p));
}
