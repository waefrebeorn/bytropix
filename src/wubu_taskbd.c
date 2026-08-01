/*
 * wubu_taskbd.c -- Task boundary detection via OOD on tok/s (BB03). C11.
 *
 * Convergence (task boundary detection 7-hop: OOD, boundary-free, RL-based):
 *   - BB03: detect task boundaries via performance divergence. When the
 *     sliding-window mean of tok/s deviates > threshold sigma from the
 *     baseline mean, declare a boundary (environment distribution shifted).
 *     At home: a tok/s drop > threshold in gen_text signals the operator
 *     has switched configs/KV strategy — the agent should consolidate EWC
 *     before sweeping new dims.
 */
#include "wubu_taskbd.h"
#include <math.h>
#include <string.h>

int wubu_taskbd_init(wubu_taskbd_t *tb, double threshold) {
    if (!tb) return -1;
    memset(tb, 0, sizeof(*tb));
    tb->capacity = WUBU_TASKBD_WINDOW;
    tb->threshold = threshold;
    return 0;
}

static double mean(const double *d, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += d[i];
    return n > 0 ? s / n : 0.0;
}

static double stddev(const double *d, int n, double m) {
    if (n < 2) return 0.0;
    double s = 0.0;
    for (int i = 0; i < n; i++) { double v = d[i] - m; s += v * v; }
    return sqrt(s / (n - 1));
}

int wubu_taskbd_observe(wubu_taskbd_t *tb, double tok_s) {
    if (!tb) return 0;
    /* Maintain sliding window */
    if (tb->n < tb->capacity) {
        tb->window[tb->n] = tok_s;
        tb->n++;
    } else {
        /* Shift window */
        memmove(tb->window, tb->window + 1, (tb->capacity - 1) * sizeof(double));
        tb->window[tb->capacity - 1] = tok_s;
    }
    /* Establish baseline from first `capacity` observations */
    if (tb->n == tb->capacity && !tb->baseline_ready) {
        tb->mean_baseline = mean(tb->window, tb->n);
        tb->baseline_ready = 1;
        return 0;
    }
    if (!tb->baseline_ready) return 0;
    /* Check divergence: |current_mean - baseline| > threshold * stddev */
    double m = mean(tb->window, tb->n);
    double sd = stddev(tb->window, tb->n, m);
    /* If window is constant (sd==0), any deviation from baseline is a boundary */
    if (sd > 0.0) {
        if (fabs(m - tb->mean_baseline) > tb->threshold * sd)
            return 1;
    } else {
        if (fabs(m - tb->mean_baseline) > 0.01)  /* baseline is ~27, this is ~5 */
            return 1;
    }
    return 0;
}

double wubu_taskbd_mean(const wubu_taskbd_t *tb) {
    if (!tb || tb->n == 0) return 0.0;
    return mean(tb->window, tb->n);
}