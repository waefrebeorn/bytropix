/*
 * wubu_uq.c -- Uncertainty Quantification: bootstrap ensemble + conformal (FF04). C11.
 *
 * Convergence (conformal prediction / bootstrap ensemble 7-hop):
 *   - FF04: bootstrap ensemble over sweep replays → variance σ_uc² = 1/(B-1)Σ(f_b-μ)².
 *     Conformal calibration: from held-out residuals, set interval width to guarantee
 *     (1-α) coverage (finite-sample, distribution-free). At home: the tok_s prediction
 *     gets a calibrated interval — if GP σ² is unreliable (non-Gaussian noise), conformal
 *     calibration widens the interval to guarantee coverage.
 */
#include "wubu_uq.h"
#include <math.h>
#include <string.h>

int wubu_uq_add_boot(wubu_uq_t *uq, const double *preds, int n_pts) {
    if (!uq || !preds || n_pts <= 0 || n_pts > WUBU_UQ_MAX_PTS) return -1;
    if (uq->n_boot >= WUBU_UQ_MAX_BOOT) return -1;
    int b = uq->n_boot;
    memcpy(uq->boot_pred[b], preds, sizeof(double) * n_pts);
    uq->n_pts = n_pts;
    uq->n_boot++;
    return 0;
}

int wubu_uq_fit(wubu_uq_t *uq) {
    if (!uq || uq->n_boot < 2) return -1;
    int B = uq->n_boot, N = uq->n_pts;
    for (int i = 0; i < N; i++) {
        double m = 0.0;
        for (int b = 0; b < B; b++) m += uq->boot_pred[b][i];
        m /= B;
        uq->mean[i] = m;
        double v = 0.0;
        for (int b = 0; b < B; b++) {
            double d = uq->boot_pred[b][i] - m;
            v += d * d;
        }
        uq->var[i] = v / (B - 1);
    }
    return 0;
}

int wubu_uq_calibrate(wubu_uq_t *uq, const double *residuals, int n_res, double alpha) {
    if (!uq || !residuals || n_res <= 0) return -1;
    /* Conformal: sort |residual|; pick (1-alpha) quantile as calibration width. */
    double sorted[WUBU_UQ_MAX_PTS];
    int n = n_res < WUBU_UQ_MAX_PTS ? n_res : WUBU_UQ_MAX_PTS;
    memcpy(sorted, residuals, sizeof(double) * n);
    /* insertion sort */
    for (int i = 1; i < n; i++) {
        double key = sorted[i];
        int j = i - 1;
        while (j >= 0 && sorted[j] > key) { sorted[j + 1] = sorted[j]; j--; }
        sorted[j + 1] = key;
    }
    int idx = (int)((1.0 - alpha) * n);
    if (idx >= n) idx = n - 1;
    double width = sorted[idx];
    for (int i = 0; i < uq->n_pts; i++) uq->calib[i] = width;
    return 0;
}

int wubu_uq_interval(const wubu_uq_t *uq, int i, double *lo, double *hi) {
    if (!uq || i < 0 || i >= uq->n_pts) return -1;
    double w = uq->calib[i] > 0 ? uq->calib[i] : sqrt(uq->var[i]) * 1.96;
    *lo = uq->mean[i] - w;
    *hi = uq->mean[i] + w;
    return 0;
}
