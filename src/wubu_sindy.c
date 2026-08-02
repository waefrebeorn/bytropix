/*
 * wubu_sindy.c -- SINDy sparse identification of nonlinear dynamics (EE02). C11.
 *
 * Convergence (SINDy / Brunton 2016 / STLSQ 7-hop):
 *   - EE02: from trajectory (x_t, dx/dt) builds a candidate library (const,
 *     x_i, x_i^2, x_i*x_j, ...) and solves sparse regression (sequential
 *     thresholded least squares) to keep only terms with significant coeff.
 *     At home: the recursive_optimize sweep produces a (config_t, tok_s_t)
 *     trajectory; SINDy discovers the *dynamical law* of how tok_s evolves.
 */
#include "wubu_sindy.h"
#include <math.h>
#include <string.h>

int wubu_sindy_build_library(const wubu_sindy_data_t *d, int *n_lib_out) {
    if (!d || !n_lib_out || d->dim <= 0) return -1;
    int dim = d->dim;
    int lib = 1 + dim + dim + (dim > 1 ? 1 : 0);  /* const + linear + quadratic + 1 cross */
    if (lib > WUBU_SINDY_MAX_LIB) return -1;
    *n_lib_out = lib;
    return 0;
}

/* Library evaluation at sample k → fills theta[lib] */
static void lib_eval(const wubu_sindy_data_t *d, int k, double *theta) {
    int dim = d->dim;
    const double *x = &d->X[k * dim];
    int t = 0;
    theta[t++] = 1.0;  /* const */
    for (int i = 0; i < dim; i++) theta[t++] = x[i];
    for (int i = 0; i < dim; i++) theta[t++] = x[i] * x[i];
    if (dim > 1) theta[t++] = x[0] * x[1];
}

/* Solve least squares per output dim via normal equations + thresholding.
 * Theta is [n_samples][lib], Y is dX[k*dim + out]. */
int wubu_sindy_fit(const wubu_sindy_data_t *d, double threshold,
                    wubu_sindy_result_t *out) {
    if (!d || !out || d->n_samples < 2) return -1;
    int dim = d->dim;
    int lib;
    if (wubu_sindy_build_library(d, &lib) != 0) return -1;
    out->n_lib = lib;
    out->threshold = threshold;

    double theta[WUBU_SINDY_MAX_LIB];
    /* For each output dimension, do STLSQ */
    for (int o = 0; o < dim; o++) {
        /* Build A = Theta^T Theta (lib×lib), b = Theta^T y (lib) */
        double A[WUBU_SINDY_MAX_LIB][WUBU_SINDY_MAX_LIB];
        double b[WUBU_SINDY_MAX_LIB];
        memset(A, 0, sizeof(A));
        memset(b, 0, sizeof(b));
        for (int k = 0; k < d->n_samples; k++) {
            lib_eval(d, k, theta);
            double yk = d->dX[k * dim + o];
            for (int i = 0; i < lib; i++) {
                b[i] += theta[i] * yk;
                for (int j = 0; j < lib; j++)
                    A[i][j] += theta[i] * theta[j];
            }
        }
        /* Simple Gauss-Seidel solver for x in A x = b (few iterations) */
        double x[WUBU_SINDY_MAX_LIB];
        memset(x, 0, sizeof(x));
        for (int iter = 0; iter < 100; iter++) {
            for (int i = 0; i < lib; i++) {
                double s = b[i];
                for (int j = 0; j < lib; j++)
                    if (j != i) s -= A[i][j] * x[j];
                if (fabs(A[i][i]) > 1e-12)
                    x[i] = s / A[i][i];
            }
        }
        /* STLSQ: threshold small coeffs to 0, re-solve */
        for (int i = 0; i < lib; i++)
            if (fabs(x[i]) < threshold) x[i] = 0.0;
        /* One more solve with active set */
        double x2[WUBU_SINDY_MAX_LIB];
        memset(x2, 0, sizeof(x2));
        for (int iter = 0; iter < 50; iter++) {
            for (int i = 0; i < lib; i++) {
                if (fabs(x[i]) < 1e-12) continue;  /* inactive */
                double s = b[i];
                for (int j = 0; j < lib; j++)
                    if (j != i && fabs(x[j]) >= 1e-12) s -= A[i][j] * x2[j];
                if (fabs(A[i][i]) > 1e-12) x2[i] = s / A[i][i];
            }
        }
        for (int i = 0; i < lib; i++) out->Xi[o][i] = x2[i];
    }
    return 0;
}

void wubu_sindy_predict(const wubu_sindy_result_t *r, const double *x, double *dx) {
    if (!r || !x || !dx) return;
    int lib = r->n_lib;
    int dim = 0;
    /* infer dim from Xi: we stored per dim, but need to know. Use max dim. */
    for (int o = 0; o < WUBU_SINDY_MAX_DIM; o++) {
        int nonzero = 0;
        for (int i = 0; i < lib; i++) if (fabs(r->Xi[o][i]) > 0) nonzero++;
        if (nonzero > 0) dim = o + 1;
    }
    for (int o = 0; o < dim; o++) {
        double acc = 0.0;
        acc += r->Xi[o][0];  /* const */
        for (int i = 0; i < dim && (1 + i) < lib; i++)
            acc += r->Xi[o][1 + i] * x[i];
        for (int i = 0; i < dim && (1 + dim + i) < lib; i++)
            acc += r->Xi[o][1 + dim + i] * x[i] * x[i];
        if (dim > 1 && (1 + 2 * dim) < lib)
            acc += r->Xi[o][1 + 2 * dim] * x[0] * x[1];
        dx[o] = acc;
    }
}
