/*
 * wubu_bo.c -- Bayesian Optimization loop (FF03). C11.
 *
 * Convergence (BO loop / sample efficiency 7-hop):
 *   - FF03: maintains a candidate set, scores each with the acquisition function
 *     over the GP surrogate, picks argmax → next config to evaluate. obj_fn is
 *     the expensive black-box (real sweep → tok_s). Repeat until convergence.
 *     At home: replaces recursive_optimize's blind 15-dim hill-climbing with a
 *     sample-efficient, uncertainty-aware optimizer.
 */
#include "wubu_bo.h"
#include "wubu_gp.h"
#include "wubu_acq.h"
#include <math.h>
#include <string.h>

int wubu_bo_select(const wubu_gp_t *gp, const wubu_acq_t *acq, wubu_bo_t *bo) {
    if (!gp || !acq || !bo || bo->n_cand == 0) return -1;
    double best_acq = -1e300;
    int best_idx = 0;
    for (int i = 0; i < bo->n_cand; i++) {
        double mean, var;
        if (wubu_gp_predict(gp, bo->cand[i], &mean, &var) != 0) continue;
        double a = wubu_acq_value(acq, mean, sqrt(var));
        bo->cand_acq[i] = a;
        if (a > best_acq) { best_acq = a; best_idx = i; }
    }
    bo->best_cand_idx = best_idx;
    bo->best_acq = best_acq;
    return best_idx;
}

int wubu_bo_step(wubu_gp_t *gp, const wubu_acq_t *acq, wubu_bo_t *bo,
                 wubu_bo_obj obj, void *ctx) {
    if (!gp || !acq || !bo || !obj) return -1;
    int idx = wubu_bo_select(gp, acq, bo);
    if (idx < 0) return -1;
    double y = obj(bo->cand[idx], bo->dim, ctx);
    wubu_gp_add(gp, bo->cand[idx], y);
    wubu_gp_fit(gp);
    return 0;
}
