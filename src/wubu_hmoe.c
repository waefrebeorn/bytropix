/* wubu_hmoe.c -- BRAIN #116 hyperbolic MoE routing.
 * See wubu_hmoe.h. */
#include "wubu_hmoe.h"
#include <string.h>
#include <math.h>

int wubu_hmoe_init(wubu_hmoe_t *m, double curvature) {
    if (!m) return -1;
    memset(m->experts, 0, sizeof m->experts);
    m->n_experts = 0;
    m->curvature = (curvature < 0) ? curvature : -1.0;
    return 0;
}

int wubu_hmoe_add_expert(wubu_hmoe_t *m, int id, const double *center) {
    if (!m || !center || m->n_experts >= WUBU_HM_MAX_EXPERTS) return -1;
    wubu_hmoe_expert_t *e = &m->experts[m->n_experts];
    e->id = id;
    for (int i = 0; i < WUBU_HM_DIM; i++) e->center[i] = center[i];
    m->n_experts++;
    return 0;
}

/* Poincare-disk hyperbolic distance. */
static double hyp_dist(const double *a, const double *b, double curv) {
    double c = -curv;
    double diff = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < WUBU_HM_DIM; i++) {
        double d = a[i] - b[i];
        diff += d * d; na += a[i]*a[i]; nb += b[i]*b[i];
    }
    double denom = (1.0 - na) * (1.0 - nb);
    if (denom <= 1e-12) denom = 1e-12;
    double arg = 1.0 + 2.0 * c * diff / denom;
    if (arg < 1.0) arg = 1.0;
    return acosh(arg) / sqrt(c);
}

int wubu_hmoe_route(const wubu_hmoe_t *m, const double *token, int k,
                    int *expert_ids, double *dists) {
    if (!m || !token || !expert_ids || k <= 0) return 0;
    /* compute distance to every expert, then selection-sort the top-k */
    double d[WUBU_HM_MAX_EXPERTS];
    int idx[WUBU_HM_MAX_EXPERTS];
    for (int i = 0; i < m->n_experts; i++) {
        d[i] = hyp_dist(token, m->experts[i].center, m->curvature);
        idx[i] = m->experts[i].id;
    }
    int emitted = 0;
    for (int pass = 0; pass < m->n_experts && emitted < k; pass++) {
        int best = pass;
        for (int j = pass + 1; j < m->n_experts; j++)
            if (d[j] < d[best]) best = j;
        if (best != pass) {
            double td = d[pass]; d[pass] = d[best]; d[best] = td;
            int ti = idx[pass]; idx[pass] = idx[best]; idx[best] = ti;
        }
        expert_ids[emitted] = idx[pass];
        if (dists) dists[emitted] = d[pass];
        emitted++;
    }
    return emitted;
}
