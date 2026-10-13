#ifndef WUBU_CANARY_H
#define WUBU_CANARY_H

/* wubu_canary.h -- OMARCHY #286 rolling-release canary gate.
 *
 * A rolling-release canary gate: a new version is deployed to a canary slice
 * first; its health (error rate, latency) is measured against the stable
 * version, and the release is promoted only if the canary is healthy. If the
 * canary regresses, the release is auto-rolled-back (fail-closed). This is the
 * progressive-delivery gate. Deterministic, real work (no stubs). */

typedef enum {
    WUBU_CANARY_PROMOTE=0,   /* canary healthy -> promote */
    WUBU_CANARY_HOLD,        /* inconclusive -> hold */
    WUBU_CANARY_ROLLBACK,    /* canary regressed -> roll back */
} wubu_canary_decision_t;

typedef struct {
    double canary_err_rate;    /* error rate observed on the canary */
    double stable_err_rate;    /* error rate on the stable version */
    double max_err_ratio;      /* canary may be at most this multiple of stable */
    int    min_samples;        /* need this many samples to decide */
    int    n_samples;
} wubu_canary_t;

/* Initialise with thresholds. */
int wubu_canary_init(wubu_canary_t *c, double max_err_ratio, int min_samples);

/* Feed a sample: 1 = canary error, 0 = canary ok. Returns 0/-1. */
int wubu_canary_sample(wubu_canary_t *c, int canary_error, int stable_error);

/* Current canary error rate. */
double wubu_canary_err_rate(const wubu_canary_t *c);

/* Make the promote/hold/rollback decision. HOLD if not enough samples. */
wubu_canary_decision_t wubu_canary_decide(const wubu_canary_t *c);

#endif
