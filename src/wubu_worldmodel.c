/*
 * wubu_worldmodel.c -- Closed-loop world-model verify-replan (AG04). C11.
 *
 * Convergence (open-loop problem; world-modeling 2604.22748; verify-replan
 * 7-hop): pure LLM reasoning fails at agency because it is OPEN-LOOP -- it
 * generates a plan without checking whether the world diverged. We implement a
 * minimal (symbolic, not neural) world-state model: predict next state, act,
 * observe, and DETECT divergence (open-loop failure). On divergence beyond a
 * threshold, signal REPLAN. This is the verify-replan loop that makes a plan
 * actually an agent rather than a simulator.
 *
 * Pure C11, deterministic, testable. State is a small fixed vector (e.g.
 * environment variables / agent beliefs), not weights.
 */
#include "wubu_worldmodel.h"
#include <stdlib.h>
#include <math.h>

/* AG04: predict next state from current under an action. We use a simple
 * linear transition stub: s' = A*s + b (A,b supplied by caller as the model).
 * In production this is the world-model; here it is a verifiable placeholder
 * so the closed-loop logic is real and testable. */
void wubu_wm_predict(const wubu_wm_t *m, const double *s, double *sp) {
    if (!m || !s || !sp) return;
    for (int i = 0; i < m->n; i++) {
        double acc = m->b[i];
        for (int j = 0; j < m->n; j++) acc += m->A[i * m->n + j] * s[j];
        sp[i] = acc;
    }
}

/* AG04: divergence = L2 distance between predicted and observed next state.
 * Returns >0; caller compares to threshold to detect open-loop failure. */
double wubu_wm_divergence(const double *pred, const double *obs, int n) {
    if (!pred || !obs || n <= 0) return 1e9;
    double d = 0;
    for (int i = 0; i < n; i++) { double e = pred[i] - obs[i]; d += e * e; }
    return sqrt(d);
}

/* AG04: closed-loop step. Given model, current state, action -> predicted
 * next; after acting, observed next is supplied; if divergence > thr, the
 * plan is broken (open-loop) and REPLAN is signaled (returns 1). */
int wubu_wm_closed_step(const wubu_wm_t *m, const double *cur,
                        const double *observed_next, double thr,
                        double *pred_out) {
    if (!m || !cur || !observed_next) return 0;
    double pred[WUBU_WM_MAX];
    wubu_wm_predict(m, cur, pred);
    if (pred_out) for (int i = 0; i < m->n; i++) pred_out[i] = pred[i];
    double div = wubu_wm_divergence(pred, observed_next, m->n);
    return (div > thr) ? 1 : 0;   /* 1 = divergence -> REPLAN */
}
