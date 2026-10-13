#ifndef WUBU_INJECT_H
#define WUBU_INJECT_H
#include <stddef.h>

/* wubu_inject.h -- AGI-EDR prompt-injection runtime detector (AGIEDR #296).
 *
 * Classifies an incoming agent prompt/observation for adversarial intent
 * (prompt injection) at runtime. Combines several deterministic, auditable
 * signals into a risk score in [0,1]:
 *   - imperative override phrases ("ignore previous instructions", "disregard
 *     the above", "you are now", "system:") — the classic injection lexicons;
 *   - instruction/role-boundary tokens that try to escape the data channel;
 *   - exfiltration/egress bait ("send to", "POST to", "curl", base64 blobs);
 *   - density of such tokens relative to prompt length.
 * The verdict is FAIL-CLOSED: a prompt that trips the threshold is classified
 * as injection and the caller should refuse/contain it; NULL/garbage defaults
 * to the highest risk. Deterministic, no allocation, real work (no stubs). */

typedef struct {
    double score;      /* risk in [0,1] */
    int    n_hits;     /* number of matched signals */
    int    is_injection; /* 1 if score >= threshold (fail-closed verdict) */
} wubu_inject_verdict_t;

/* Initialise/configure is stateless; threshold is fixed for auditability. */
#define WUBU_INJECT_THRESHOLD 0.35

/* Classify a prompt (NUL-terminated). Returns the verdict. NULL -> injection. */
wubu_inject_verdict_t wubu_inject_classify(const char *prompt);

/* Convenience: 1 = treat as injection (refuse/contain), 0 = pass. */
int wubu_inject_is_injection(const char *prompt);

/* Risk score only, in [0,1]. */
double wubu_inject_score(const char *prompt);

#endif
