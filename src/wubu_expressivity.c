/* wubu_expressivity.c -- PATHS #324 expressivity-invariant risk tracking.
 * See wubu_expressivity.h. */
#include "wubu_expressivity.h"
#include <math.h>

int wubu_expr_init(wubu_expressivity_t *x, double baseline, double tolerance) {
    if (!x) return -1;
    if (tolerance < 0) tolerance = 0;
    x->baseline = baseline;
    x->tolerance = tolerance;
    x->n_mutations = 0;
    x->n_risky = 0;
    x->n_novel = 0;
    return 0;
}

wubu_expr_risk_t wubu_expr_assess(wubu_expressivity_t *x, double new_expressivity) {
    if (!x) return WUBU_EXPR_RISKY;   /* fail-closed */
    x->n_mutations++;
    double rel;
    if (x->baseline > 0) rel = (new_expressivity - x->baseline) / x->baseline;
    else rel = new_expressivity > 0 ? 1.0 : 0.0;
    wubu_expr_risk_t r;
    if (rel < -x->tolerance) { r = WUBU_EXPR_RISKY; x->n_risky++; }
    else if (rel > x->tolerance) { r = WUBU_EXPR_NOVEL; x->n_novel++; }
    else r = WUBU_EXPR_SAFE;
    return r;
}

int wubu_expr_accept(wubu_expressivity_t *x, double new_expressivity) {
    if (!x) return -1;
    x->baseline = new_expressivity;
    return 0;
}

double wubu_expr_risk_rate(const wubu_expressivity_t *x) {
    if (!x || x->n_mutations == 0) return 0.0;
    return (double)x->n_risky / (double)x->n_mutations;
}
