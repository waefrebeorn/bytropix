#ifndef WUBU_PROMOTE_H
#define WUBU_PROMOTE_H
#include <stdint.h>

/* wubu_promote.h -- AGENT #147 independent verifier gates every promotion
 * + #150 trace span emitted on every promote.
 *
 * The promotion gate for the self-improvement loop. A mutation is promoted
 * (accepted into the archive) ONLY if an INDEPENDENT verifier (a separate
 * process/agent, not the proposer) attests it passed -- the proposer cannot
 * self-approve. On a successful promotion, a trace span is emitted (trace_id,
 * span_id, parent_id, proposal, verifier id, status) for observability. Every
 * promotion and rejection is recorded in the lineage. Fail-closed: no
 * independent verifier attestation -> reject. Deterministic, no allocation,
 * real work (no stubs). */

#define WUBU_PROMOTE_MAX 256
#define WUBU_PROMOTE_IDLEN 96

typedef enum { WUBU_PROMOTE_PENDING=0, WUBU_PROMOTE_ACCEPTED, WUBU_PROMOTE_REJECTED } wubu_promote_status_t;

typedef struct {
    char     proposal_id[WUBU_PROMOTE_IDLEN];
    char     proposer_id[WUBU_PROMOTE_IDLEN];
    char     verifier_id[WUBU_PROMOTE_IDLEN];
    int      verifier_attested;   /* 1 = independent verifier approved */
    wubu_promote_status_t status;
    uint64_t trace_id;
    uint64_t span_id;
} wubu_promote_rec_t;

typedef struct {
    wubu_promote_rec_t recs[WUBU_PROMOTE_MAX];
    int  n;
    uint64_t next_span_id;
    int  n_accepted;
    int  n_rejected;
} wubu_promote_t;

/* Initialise the promotion gate. */
int wubu_promote_init(wubu_promote_t *p);

/* Propose a promotion: records a pending proposal. Returns its index or -1. */
int wubu_promote_propose(wubu_promote_t *p, const char *proposal_id,
                         const char *proposer_id, uint64_t trace_id);

/* Gate the promotion with an independent verifier's attestation. The
 * verifier_id MUST differ from the proposer_id (independence enforced) and
 * attested must be 1. On success the promotion is accepted and a trace span
 * is emitted (span_id returned); on failure it is rejected. Returns the
 * span_id (>0) if accepted, 0 if rejected, -1 on bad input. Fail-closed. */
uint64_t wubu_promote_gate(wubu_promote_t *p, int idx,
                           const char *verifier_id, int attested);

/* Look up a promotion record by index. */
const wubu_promote_rec_t *wubu_promote_get(const wubu_promote_t *p, int idx);

/* Stats. */
int wubu_promote_accepted(const wubu_promote_t *p);
int wubu_promote_rejected(const wubu_promote_t *p);

#endif
