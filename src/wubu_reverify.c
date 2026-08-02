/*
 * wubu_reverify.c -- Closed-loop self-verification (EE07). C11.
 */
#include "wubu_reverify.h"
#include <string.h>

int wubu_reverify_init(wubu_reverify_t *r, double shift_thresh,
                       double fit_thresh)
{
    if (!r || shift_thresh <= 0 || fit_thresh < 0) return -1;
    memset(r, 0, sizeof(*r));
    r->shift_thresh = shift_thresh;
    r->fit_thresh   = fit_thresh;
    return 0;
}

int wubu_reverify_step(wubu_reverify_t *r, double divergence,
                       const double *fit, int n_inv, double fresh_fit,
                       uint32_t epoch)
{
    if (!r || !fit || n_inv <= 0 || n_inv > WUBU_RV_MAX_INV) return 0;
    if (epoch > r->epoch) r->epoch = epoch;
    memcpy(r->fit, fit, (size_t)n_inv * sizeof(double));

    int triggered = 0;
    if (divergence > r->shift_thresh &&
        epoch > r->last_verify_epoch) {
        /* shift detected: re-verify. The caller re-synthesizes and
         * passes the fresh synthesis fit in the NEXT step; we mark the
         * epoch so a re-verification happens at most once per epoch. */
        triggered = 1;
        r->triggers++;
        r->last_verify_epoch = epoch;
    }

    /* replacement pass: any invariant whose fit degraded past the
     * threshold is replaced by the fresh synthesis fit (>= 0 means the
     * caller re-ran discovery this epoch). */
    if (fresh_fit >= 0) {
        for (int i = 0; i < n_inv; i++) {
            if (r->fit[i] > r->fit_thresh) {
                r->fit[i] = fresh_fit;
                r->replaced[i]++;
                r->replacements++;
            }
        }
    }
    return triggered;
}
