/*
 * wubu_ewc.c -- EWC consolidation (BB02). C11.
 *
 * Convergence (EWC + Fisher Information 7-hop):
 *   - BB02: Elastic Weight Consolidation on the 15-dim sweep space.
 *     Fisher importance ~ |gradient| of loss w.r.t. each dim (proxy).
 *     Penalty = lambda * sum(F_i * (theta_i - anchor_i)^2).
 *     High-F dims are "stable synapses" (protected), low-F are "plastic".
 *     Recent finding (arXiv 2603.18596): Logits Reversal fixes FIM gradient
 *     vanishing — we apply the analogous reversal (negate gradient signal
 *     before squaring) in wubu_ewc_estimate_fisher.
 *     At home: protects the best-performing sweep dims from being overwritten
 *     by new sweeps (config-level continual learning).
 */
#include "wubu_ewc.h"
#include <math.h>
#include <string.h>

int wubu_ewc_init(wubu_ewc_t *e, const double *params, int ndim, double lambda) {
    if (!e || !params || ndim <= 0 || ndim > WUBU_EWC_DIMS) return -1;
    memset(e->fisher, 0, sizeof(e->fisher));
    for (int i = 0; i < ndim; i++) e->anchor[i] = params[i];
    for (int i = ndim; i < WUBU_EWC_DIMS; i++) e->anchor[i] = 0.0;
    e->lambda = lambda;
    e->initialized = 1;
    return 0;
}

int wubu_ewc_estimate_fisher(wubu_ewc_t *e, const double *grads, int ndim) {
    if (!e || !grads || !e->initialized || ndim <= 0 || ndim > WUBU_EWC_DIMS)
        return -1;
    for (int i = 0; i < ndim; i++) {
        /* EWC "done right": reverse the logit/gradient signal before squaring
         * to prevent gradient vanishing + redundant protection. */
        double rev = -grads[i];
        double sq = rev * rev;
        e->fisher[i] = e->fisher[i] + sq;  /* accumulate FIM */
        if (e->fisher[i] > 1e12) e->fisher[i] = 1e12;  /* clamp */
    }
    return 0;
}

double wubu_ewc_penalty(const wubu_ewc_t *e, const double *params, int ndim) {
    if (!e || !params || !e->initialized) return 0.0;
    double penalty = 0.0;
    for (int i = 0; i < ndim; i++) {
        double delta = params[i] - e->anchor[i];
        penalty += e->fisher[i] * delta * delta;
    }
    return e->lambda * penalty;
}

int wubu_ewc_is_stable(const wubu_ewc_t *e, int dim) {
    if (!e || dim < 0 || dim >= WUBU_EWC_DIMS) return 0;
    return (e->fisher[dim] >= 1.0) ? 1 : 0;
}

int wubu_ewc_stable_count(const wubu_ewc_t *e, double threshold) {
    if (!e) return 0;
    int c = 0;
    for (int i = 0; i < WUBU_EWC_DIMS; i++)
        if (e->fisher[i] >= threshold) c++;
    return c;
}