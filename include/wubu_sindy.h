/*
 * wubu_sindy.h -- SINDy: sparse identification of nonlinear dynamics (EE02).
 */
#ifndef WUBU_SINDY_H
#define WUBU_SINDY_H

#define WUBU_SINDY_MAX_LIB 20  /* library terms */
#define WUBU_SINDY_MAX_DIM 8

typedef struct {
    int n_samples;  /* trajectory length */
    int dim;        /* state dimension */
    const double *X;  /* [n_samples][dim] state */
    const double *dX; /* [n_samples][dim] derivative */
} wubu_sindy_data_t;

typedef struct {
    /* Xi[dim][lib]: sparse coefficient matrix (governing equations) */
    double Xi[WUBU_SINDY_MAX_DIM][WUBU_SINDY_MAX_LIB];
    int    n_lib;
    double threshold;  /* STLSQ threshold */
} wubu_sindy_result_t;

/* Library terms: 0=const, 1..dim=x_i, dim+1..2dim=x_i^2, 2dim+1=x1*x2, ... */
int wubu_sindy_build_library(const wubu_sindy_data_t *d, int *n_lib_out);
/* Sequential thresholded least squares (STLSQ) sparse regression. */
int wubu_sindy_fit(const wubu_sindy_data_t *d, double threshold,
                    wubu_sindy_result_t *out);
/* Predict derivative from discovered Xi. */
void wubu_sindy_predict(const wubu_sindy_result_t *r, const double *x, double *dx);

#endif