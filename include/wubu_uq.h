/*
 * wubu_uq.h -- Uncertainty Quantification (bootstrap ensemble + conformal) (FF04).
 */
#ifndef WUBU_UQ_H
#define WUBU_UQ_H

#define WUBU_UQ_MAX_BOOT 32
#define WUBU_UQ_MAX_PTS 128

typedef struct {
    int n_boot;
    int n_pts;
    double boot_pred[WUBU_UQ_MAX_BOOT][WUBU_UQ_MAX_PTS];  /* bootstrap replicates */
    double mean[WUBU_UQ_MAX_PTS];
    double var[WUBU_UQ_MAX_PTS];     /* bootstrap variance */
    double calib[WUBU_UQ_MAX_PTS];   /* conformal calibration width */
} wubu_uq_t;

/* Add a bootstrap replicate prediction set (n_pts values). */
int wubu_uq_add_boot(wubu_uq_t *uq, const double *preds, int n_pts);
/* Compute mean + bootstrap variance across replicates. */
int wubu_uq_fit(wubu_uq_t *uq);
/* Conformal calibration: given held-out residuals, set calibration width
   to achieve (1-alpha) coverage. */
int wubu_uq_calibrate(wubu_uq_t *uq, const double *residuals, int n_res, double alpha);
/* Prediction interval at point i: [mean - width, mean + width]. */
int wubu_uq_interval(const wubu_uq_t *uq, int i, double *lo, double *hi);

#endif