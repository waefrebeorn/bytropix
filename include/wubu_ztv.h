#ifndef WUBU_ZTV_H
#define WUBU_ZTV_H

/* wubu_ztverify.h -- NAMESPACE #193 zero-trust continuous verification.
 *
 * Zero-trust: never trust a capability once granted -- RE-VERIFY on every
 * action. Each action presents a capability token; the gate re-checks it
 * against the current revocation state and a freshness (nonce/epoch) on every
 * single request. A revoked or stale capability is denied even if it was valid
 * when issued. This is per-action continuous verification, not one-time auth.
 * Deterministic, real work (no stubs). */

#define WUBU_ZT_MAX_CAPS 32

typedef struct {
    int cap_id;
    int revoked;          /* 1 = revoked */
    unsigned long long epoch;   /* freshness epoch when issued */
} wubu_zt_cap_t;

typedef struct {
    wubu_zt_cap_t caps[WUBU_ZT_MAX_CAPS];
    int n;
    unsigned long long current_epoch;   /* bumped on any revocation */
    int n_allowed;
    int n_denied;
} wubu_ztv_t;

/* Initialise. */
int wubu_ztv_init(wubu_ztv_t *z);

/* Issue a capability (id) at the current epoch. Returns 0/-1. */
int wubu_ztv_issue(wubu_ztv_t *z, int cap_id);

/* Revoke a capability. Bumps the global epoch (invalidating freshness).
 * Returns 0/-1. */
int wubu_ztv_revoke(wubu_ztv_t *z, int cap_id);

/* Verify an action: the cap must exist, not be revoked, and be FRESH
 * (its epoch == the current epoch). Returns 1 if the action is allowed,
 * 0 if denied (revoked/stale/unknown). Every call re-verifies. */
int wubu_ztv_verify(wubu_ztv_t *z, int cap_id);

/* Telemetry: allowed / denied counts. */
void wubu_ztv_stats(const wubu_ztv_t *z, int *allowed, int *denied);

#endif
