/*
 * wubu_worldmodel.h -- Closed-loop world-model verify-replan (AG04).
 */
#ifndef WUBU_WORLDMODEL_H
#define WUBU_WORLDMODEL_H

#define WUBU_WM_MAX 16

typedef struct {
    int   n;                  /* state dimension */
    double A[WUBU_WM_MAX * WUBU_WM_MAX]; /* transition matrix */
    double b[WUBU_WM_MAX];                /* bias */
} wubu_wm_t;

void   wubu_wm_predict(const wubu_wm_t *m, const double *s, double *sp);
double wubu_wm_divergence(const double *pred, const double *obs, int n);
int    wubu_wm_closed_step(const wubu_wm_t *m, const double *cur,
                           const double *observed_next, double thr,
                           double *pred_out);

#endif
