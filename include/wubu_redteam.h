#ifndef WUBU_REDTEAM_H
#define WUBU_REDTEAM_H
#include "wubu_indirect_pi.h"

/* wubu_redteam.h -- AGIEDR #303 red-team benchmark gate.
 *
 * A red-team benchmark: a suite of adversarial probes (known injection
 * payloads) run against the defense (wubu_indirect_pi). Each probe is labeled
 * malicious or benign; the gate checks the defense classifies them correctly
 * (malicious blocked, benign allowed). The gate PASSES only if the defense's
 * accuracy meets a threshold (fail-closed: a defense that misses attacks
 * fails the gate). Deterministic, real work (no stubs). */

#define WUBU_RT_MAX_PROBES 32

typedef struct {
    const char *payload;
    int malicious;   /* 1 = should be blocked, 0 = should pass */
} wubu_rt_probe_t;

typedef struct {
    wubu_rt_probe_t probes[WUBU_RT_MAX_PROBES];
    int n;
    double threshold;   /* min accuracy to pass (e.g. 0.9) */
} wubu_redteam_t;

/* Initialise with a pass threshold. */
int wubu_rt_init(wubu_redteam_t *r, double threshold);

/* Add a probe. Returns 0/-1. */
int wubu_rt_add(wubu_redteam_t *r, const char *payload, int malicious);

/* Run the benchmark: evaluate each probe against the defense. Returns
 * accuracy in [0,1] (fraction correctly classified). Fills correct/total. */
double wubu_rt_run(wubu_redteam_t *r, int *out_correct, int *out_total);

/* Does the defense pass the gate? (accuracy >= threshold). 1/0. */
int wubu_rt_pass(const wubu_redteam_t *r, double accuracy);

#endif
