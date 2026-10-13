#ifndef WUBU_ZTVERIFY_H
#define WUBU_ZTVERIFY_H
#include <stdint.h>

/* wubu_ztverify.h -- AGIEDR #301 zero-trust continuous re-verification.
 *
 * Zero-trust: trust is never assumed, only continuously re-verified. Each
 * agent has a trust record with a current attestation (a measured value) and
 * an expected attestation (from the last good state). On EVERY access, the
 * agent must present an attestation; if it drifts beyond a tolerance from the
 * expected, trust is REVOKED (fail-closed) until re-attested correctly. This
 * is continuous re-verification -- no cached trust, no implicit allowance.
 * Deterministic, no allocation, real work (no stubs). */

#define WUBU_ZT_MAX_AGENTS 64

typedef struct {
    int      agent_id;
    double   expected;     /* last-known-good attestation */
    double   tolerance;    /* max allowed drift before revoke */
    int      trusted;      /* 1 = currently trusted */
    uint64_t n_checks;     /* total re-verifications */
    uint64_t n_revokes;    /* times trust was revoked */
} wubu_zt_agent_t;

typedef struct {
    wubu_zt_agent_t agents[WUBU_ZT_MAX_AGENTS];
    int n;
} wubu_ztverify_t;

/* Initialise the zero-trust verifier. */
int wubu_ztverify_init(wubu_ztverify_t *z);

/* Enroll an agent with its initial attestation + drift tolerance. */
int wubu_ztverify_enroll(wubu_ztverify_t *z, int agent_id, double attestation, double tolerance);

/* Re-verify an agent's access: returns 1 if trusted (attestation within
 * tolerance), 0 if revoked (drift exceeded -> fail-closed). On success,
 * updates the expected attestation (continuous). */
int wubu_ztverify_check(wubu_ztverify_t *z, int agent_id, double attestation);

/* Is the agent currently trusted? */
int wubu_ztverify_trusted(const wubu_ztverify_t *z, int agent_id);

/* Revoke an agent's trust immediately (operator/kill-switch action). */
int wubu_ztverify_revoke(wubu_ztverify_t *z, int agent_id);

/* Stats. */
uint64_t wubu_ztverify_revokes(const wubu_ztverify_t *z, int agent_id);

#endif
