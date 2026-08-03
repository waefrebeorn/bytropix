/* wubu_rollout.c -- the Balanced Adaptive Rollout allocation. */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_rollout.h"

int wubu_rollout_alloc(const float *succ, int n, int budget,
                       float gamma, int *out)
{
    if (!succ || !out || n < 1 || budget < 1 || gamma < 0) return 0;
    /* the difficulty weights: (1 - succ + eps)^gamma, the unknown = 0.5 */
    double *w = (double *)malloc((size_t)n * sizeof(double));
    if (!w) return 0;
    double wsum = 0;
    for (int i = 0; i < n; i++) {
        double fail = (succ[i] < 0) ? 0.5 : (1.0 - (succ[i] > 1 ? 1 : succ[i]));
        w[i] = pow(fail + 1e-6, gamma);
        wsum += w[i];
    }
    if (wsum <= 0) { free(w); return 0; }
    /* the largest-remainder allocation (the counts sum EXACTLY to budget) */
    double *raw = (double *)malloc((size_t)n * sizeof(double));
    if (!raw) { free(w); return 0; }
    for (int i = 0; i < n; i++) raw[i] = w[i] / wsum * budget;
    int given = 0;
    for (int i = 0; i < n; i++) { out[i] = (int)raw[i]; given += out[i]; }
    int rem = budget - given;
    for (int r = 0; r < rem; r++) {
        int best = 0;
        double bestfrac = -1;
        for (int i = 0; i < n; i++) {
            double frac = raw[i] - out[i];
            if (frac > bestfrac) { bestfrac = frac; best = i; }
        }
        out[best]++;
    }
    free(raw);
    free(w);
    return 1;
}
