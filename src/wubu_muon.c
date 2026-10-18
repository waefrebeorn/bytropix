/*
 * wubu_muon.c -- public Newton-Schulz polar orthogonalization (C11).
 *
 * Why this exists separately from the training path:
 * ns5_inplace() in wubu_backprop.c renormalizes to unit Frobenius norm on
 * every step. That bounds the matrix but pins its spectral norm at
 * 1/sqrt(n), which keeps the Muon quintic outside the basin where it would
 * drive singular values to 1. Measured on a 64x64 Gaussian, that routine
 * leaves ||X^T X - I||_F ~ 7.9 and never improves.
 *
 * This entry point applies the Frobenius normalization ONCE (so sigma_max<=1)
 * and then the quadratic NS map X <- 0.5 X (3I - X^T X), which converges to
 * the orthogonal polar factor. The iteration count is explicit: recovering
 * from a small sigma needs more than the 5 steps the quintic advertises.
 */
#include "wubu_muon.h"
#include <stdlib.h>
#include <math.h>

int wubu_muon_orthogonalize(float *x, int rows, int cols, int iters)
{
    if (!x || rows <= 0 || cols <= 0 || iters <= 0) return -1;

    int tr = rows < cols;
    int r = tr ? cols : rows;   /* tall */
    int c = tr ? rows : cols;   /* small */

    float *w = (float *)malloc((size_t)rows * (size_t)cols * sizeof(float));
    float *g = (float *)malloc((size_t)c * (size_t)c * sizeof(float));
    if (!w || !g) { free(w); free(g); return -1; }

    float *X = x;
    if (tr) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                w[(size_t)j * (size_t)rows + (size_t)i] =
                    x[(size_t)i * (size_t)cols + (size_t)j];
        X = w;
    }

    double ss = 0.0;
    for (int i = 0; i < r * c; i++) ss += (double)X[i] * X[i];
    double nrm = sqrt(ss);
    if (nrm < 1e-12) { free(w); free(g); return 0; }   /* zero stays zero */
    float inv = (float)(1.0 / nrm);
    for (int i = 0; i < r * c; i++) X[i] *= inv;

    for (int it = 0; it < iters; it++) {
        /* G = X^T X  (c x c) */
        for (int a = 0; a < c; a++) {
            for (int b = 0; b < c; b++) {
                double s = 0.0;
                for (int k = 0; k < r; k++)
                    s += (double)X[(size_t)k * (size_t)c + (size_t)a] *
                         (double)X[(size_t)k * (size_t)c + (size_t)b];
                g[(size_t)a * (size_t)c + (size_t)b] = (float)s;
            }
        }
        /* X <- 0.5 * (3X - X G) */
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                double s = 0.0;
                for (int k = 0; k < c; k++)
                    s += (double)X[(size_t)i * (size_t)c + (size_t)k] *
                         (double)g[(size_t)k * (size_t)c + (size_t)j];
                X[(size_t)i * (size_t)c + (size_t)j] =
                    (float)(0.5 * (3.0 * (double)X[(size_t)i * (size_t)c + (size_t)j] - s));
            }
        }
    }

    if (tr) {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                x[(size_t)i * (size_t)cols + (size_t)j] =
                    w[(size_t)j * (size_t)rows + (size_t)i];
    }
    free(w);
    free(g);
    return 0;
}