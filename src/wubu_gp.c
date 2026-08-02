/*
 * wubu_gp.c -- Gaussian Process regression (FF01). C11.
 *
 * Convergence (GP regression / RBF kernel / Cholesky 7-hop):
 *   - FF01: RBF kernel k(x,x') = σ²_f exp(-||x-x'||²/(2ℓ²)) + noise·δ.
 *     Fit = Cholesky decomposition of (K + noise·I) → L; alpha = L^T \ (L \ y).
 *     Predict: mean = k_*^T alpha; var = k(x,x) - k_*^T (L^T \ (L \ k_*)).
 *     At home: surrogate over recursive_optimize's 15-dim sweep space; μ predicts
 *     tok_s, σ² quantifies where the sweep is unsure (guides acquisition).
 */
#include "wubu_gp.h"
#include <math.h>
#include <string.h>

static double rbf(const wubu_gp_t *gp, const double *a, const double *b) {
    double d2 = 0.0;
    for (int i = 0; i < gp->dim; i++) {
        double d = a[i] - b[i];
        d2 += d * d;
    }
    return gp->sigma2_f * exp(-d2 / (2.0 * gp->length_scale * gp->length_scale));
}

int wubu_gp_add(wubu_gp_t *gp, const double *x, double y) {
    if (!gp || gp->n >= WUBU_GP_MAX_PTS) return -1;
    memcpy(gp->X[gp->n], x, sizeof(double) * gp->dim);
    gp->y[gp->n] = y;
    gp->n++;
    gp->fitted = 0;
    return 0;
}

int wubu_gp_fit(wubu_gp_t *gp) {
    if (!gp || gp->n == 0) return -1;
    int n = gp->n;
    double K[WUBU_GP_MAX_PTS][WUBU_GP_MAX_PTS];
    memset(K, 0, sizeof(K));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double k = rbf(gp, gp->X[i], gp->X[j]);
            if (i == j) k += gp->noise * gp->noise;
            K[i][j] = k;
        }
    }
    /* Cholesky: L L^T = K */
    double L[WUBU_GP_MAX_PTS][WUBU_GP_MAX_PTS];
    memset(L, 0, sizeof(L));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = K[i][j];
            for (int k = 0; k < j; k++) s -= L[i][k] * L[j][k];
            if (i == j) {
                if (s <= 0) s = 1e-8;  /* jitter for PSD */
                L[i][j] = sqrt(s);
            } else {
                L[i][j] = s / L[j][j];
            }
        }
    }
    /* alpha = L^T \ (L \ y) */
    double tmp[WUBU_GP_MAX_PTS];
    for (int i = 0; i < n; i++) {
        double s = gp->y[i];
        for (int k = 0; k < i; k++) s -= L[i][k] * tmp[k];
        tmp[i] = s / L[i][i];
    }
    for (int i = n - 1; i >= 0; i--) {
        double s = tmp[i];
        for (int k = i + 1; k < n; k++) s -= L[k][i] * gp->alpha[k];
        gp->alpha[i] = s / L[i][i];
    }
    memcpy(gp->L, L, sizeof(L));
    gp->fitted = 1;
    return 0;
}

int wubu_gp_predict(const wubu_gp_t *gp, const double *x, double *mean, double *var) {
    if (!gp || gp->n == 0 || !gp->fitted) return -1;
    int n = gp->n;
    double ks[WUBU_GP_MAX_PTS];  /* k_*: cov(x, X_i) */
    for (int i = 0; i < n; i++) ks[i] = rbf(gp, x, gp->X[i]);
    /* mean = ks^T alpha */
    double m = 0.0;
    for (int i = 0; i < n; i++) m += ks[i] * gp->alpha[i];
    /* var = k(x,x) - ks^T (L^T \ (L \ ks)) */
    double v = gp->sigma2_f;
    double tmp[WUBU_GP_MAX_PTS];
    for (int i = 0; i < n; i++) {
        double s = ks[i];
        for (int k = 0; k < i; k++) s -= gp->L[i][k] * tmp[k];
        tmp[i] = s / gp->L[i][i];
    }
    double v2 = 0.0;
    for (int i = 0; i < n; i++) v2 += tmp[i] * tmp[i];
    v -= v2;
    if (v < 0) v = 0;  /* numerical */
    *mean = m; *var = v;
    return 0;
}
