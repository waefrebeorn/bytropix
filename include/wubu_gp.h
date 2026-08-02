/*
 * wubu_gp.h -- Gaussian Process regression (surrogate model) (FF01).
 */
#ifndef WUBU_GP_H
#define WUBU_GP_H

#define WUBU_GP_MAX_PTS 128
#define WUBU_GP_MAX_DIM 15

typedef struct {
    int n;                       /* number of observations */
    int dim;                     /* input dimension */
    double X[WUBU_GP_MAX_PTS][WUBU_GP_MAX_DIM];
    double y[WUBU_GP_MAX_PTS];
    double sigma2_f;             /* signal variance */
    double length_scale;         /* RBF length scale */
    double noise;                /* observation noise */
    /* Cholesky L of (K + noise*I) for prediction */
    double L[WUBU_GP_MAX_PTS][WUBU_GP_MAX_PTS];
    double alpha[WUBU_GP_MAX_PTS];  /* K^-1 y */
    int fitted;
} wubu_gp_t;

/* Add a point and refit (recompute Cholesky + alpha). */
int  wubu_gp_add(wubu_gp_t *gp, const double *x, double y);
int  wubu_gp_fit(wubu_gp_t *gp);
/* Predict at x: fills mean and var (variance of predictive distribution). */
int  wubu_gp_predict(const wubu_gp_t *gp, const double *x, double *mean, double *var);

#endif