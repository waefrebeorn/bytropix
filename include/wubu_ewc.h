/*
 * wubu_ewc.h -- EWC consolidation (BB02). C11.
 */
#ifndef WUBU_EWC_H
#define WUBU_EWC_H

#define WUBU_EWC_DIMS 15  /* matches recursive_optimize N_DIM */

typedef struct {
    double fisher[WUBU_EWC_DIMS];    /* Fisher importance per dim */
    double anchor[WUBU_EWC_DIMS];    /* reference params (consolidation point) */
    double lambda;     /* regularization strength */
    int    initialized;
} wubu_ewc_t;

/* Initialize EWC with current params as anchor (first task). */
int  wubu_ewc_init(wubu_ewc_t *e, const double *params, int ndim, double lambda);
/* Estimate Fisher importance from a gradient/loss signal. */
int  wubu_ewc_estimate_fisher(wubu_ewc_t *e, const double *grads, int ndim);
/* Compute the EWC penalty: lambda * sum(F_i * (theta_i - anchor_i)^2). */
double wubu_ewc_penalty(const wubu_ewc_t *e, const double *params, int ndim);
/* Check if dim is "stable" (protected) — Fisher importance above threshold. */
int  wubu_ewc_is_stable(const wubu_ewc_t *e, int dim);
/* How many dims are stable (protected from change)? */
int  wubu_ewc_stable_count(const wubu_ewc_t *e, double threshold);

#endif