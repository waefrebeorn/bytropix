#ifndef WUBU_CAPREVOKE_H
#define WUBU_CAPREVOKE_H

/* wubu_caprevoke.h -- NAMESPACE #191 capability revocation propagates.
 *
 * A capability tree: each capability may derive children (attenuated
 * capabilities). Revoking a capability must IMMEDIATELY detach it and
 * transitively revoke every descendant -- revocation propagates down the tree.
 * A detached capability no longer authorizes anything, and no orphaned child
 * survives its parent. Deterministic, real work (no stubs). */

#define WUBU_CR_MAX 64

typedef struct {
    int id;
    int parent;         /* -1 = root */
    int revoked;        /* 1 = revoked/detached */
    int rights;         /* bitmask of rights (attenuated copies have fewer) */
} wubu_cr_cap_t;

typedef struct {
    wubu_cr_cap_t caps[WUBU_CR_MAX];
    int n;
    int n_revocations;  /* total caps revoked via propagation */
} wubu_caprevoke_t;

/* Initialise. */
int wubu_cr_init(wubu_caprevoke_t *c);

/* Create a capability. parent = -1 for a root, else a parent id. rights are
 * attenuated to be a subset of the parent's. Returns the new cap id or -1. */
int wubu_cr_create(wubu_caprevoke_t *c, int parent, int rights);

/* Revoke a capability AND all its descendants (cascade). Returns the number of
 * capabilities revoked (including itself), or -1 if the id is unknown/already
 * revoked. */
int wubu_cr_revoke(wubu_caprevoke_t *c, int id);

/* Is a capability currently valid (exists and not revoked)? 1/0. */
int wubu_cr_valid(const wubu_caprevoke_t *c, int id);

/* Rights of a capability (0 if revoked/unknown). */
int wubu_cr_rights(const wubu_caprevoke_t *c, int id);

#endif
