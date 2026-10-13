/* wubu_canary.c -- OMARCHY #286 rolling-release canary gate.
 * See wubu_canary.h. */
#include "wubu_canary.h"
#include <stdint.h>

int wubu_canary_init(wubu_canary_t *c, double max_err_ratio, int min_samples) {
    if (!c) return -1;
    if (max_err_ratio < 1.0) max_err_ratio = 1.0;   /* canary must not be worse */
    if (min_samples < 1) min_samples = 1;
    c->canary_err_rate = 0.0;
    c->stable_err_rate = 0.0;
    c->max_err_ratio = max_err_ratio;
    c->min_samples = min_samples;
    c->n_samples = 0;
    /* running sums via rate * n: keep rates as weighted means */
    return 0;
}

int wubu_canary_sample(wubu_canary_t *c, int canary_error, int stable_error) {
    if (!c) return -1;
    int n = c->n_samples;
    /* incremental weighted mean of the error rates */
    c->canary_err_rate += ((canary_error ? 1.0 : 0.0) - c->canary_err_rate) / (n + 1);
    c->stable_err_rate += ((stable_error ? 1.0 : 0.0) - c->stable_err_rate) / (n + 1);
    c->n_samples++;
    return 0;
}

double wubu_canary_err_rate(const wubu_canary_t *c) {
    return c ? c->canary_err_rate : 0.0;
}

wubu_canary_decision_t wubu_canary_decide(const wubu_canary_t *c) {
    if (!c) return WUBU_CANARY_ROLLBACK;   /* fail-closed */
    if (c->n_samples < c->min_samples) return WUBU_CANARY_HOLD;
    /* rollback if the canary error rate exceeds the allowed ratio of stable */
    if (c->stable_err_rate > 0.0) {
        if (c->canary_err_rate > c->stable_err_rate * c->max_err_ratio)
            return WUBU_CANARY_ROLLBACK;
    } else {
        /* stable is clean: any canary error rate above the (implicit) zero
         * tolerance of max_err_ratio is a rollback */
        if (c->canary_err_rate > 0.0)
            return WUBU_CANARY_ROLLBACK;
    }
    return WUBU_CANARY_PROMOTE;
}
