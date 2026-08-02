/*
 * wubu_bo.h -- Bayesian Optimization loop (FF03).
 */
#ifndef WUBU_BO_H
#define WUBU_BO_H

#include "wubu_gp.h"
#include "wubu_acq.h"
#define WUBU_BO_MAX_ITERS 64
#define WUBU_BO_MAX_CAND 256

typedef struct {
    int   dim;
    int   n_cand;                       /* candidate configs to consider */
    double cand[WUBU_BO_MAX_CAND][WUBU_GP_MAX_DIM];
    double cand_acq[WUBU_BO_MAX_CAND];  /* acquisition value per candidate */
    int   best_cand_idx;                /* argmax acquisition */
    double best_acq;
} wubu_bo_t;

/* Given GP + acquisition, evaluate acquisition over all candidates,
   return the index of the best next config to evaluate. */
int wubu_bo_select(const wubu_gp_t *gp, const wubu_acq_t *acq,
                    wubu_bo_t *bo);

/* One BO step: pick candidate, "observe" (callback fills y), update GP.
   obj_fn is the black-box (e.g. real sweep). Returns 0 ok. */
typedef double (*wubu_bo_obj)(const double *x, int dim, void *ctx);
int wubu_bo_step(wubu_gp_t *gp, const wubu_acq_t *acq, wubu_bo_t *bo,
                 wubu_bo_obj obj, void *ctx);

#endif