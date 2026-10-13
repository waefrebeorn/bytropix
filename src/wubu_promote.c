/* wubu_promote.c -- AGENT #147 independent verifier gates every promotion
 * + #150 trace span emitted on every promote. See wubu_promote.h. */
#include "wubu_promote.h"
#include <string.h>

int wubu_promote_init(wubu_promote_t *p) {
    if (!p) return -1;
    memset(p->recs, 0, sizeof p->recs);
    p->n = 0;
    p->next_span_id = 1;
    p->n_accepted = 0;
    p->n_rejected = 0;
    return 0;
}

int wubu_promote_propose(wubu_promote_t *p, const char *proposal_id,
                         const char *proposer_id, uint64_t trace_id) {
    if (!p || !proposal_id || !proposer_id) return -1;
    if (p->n >= WUBU_PROMOTE_MAX) return -1;
    wubu_promote_rec_t *r = &p->recs[p->n];
    memset(r, 0, sizeof *r);
    strncpy(r->proposal_id, proposal_id, WUBU_PROMOTE_IDLEN - 1);
    strncpy(r->proposer_id, proposer_id, WUBU_PROMOTE_IDLEN - 1);
    r->status = WUBU_PROMOTE_PENDING;
    r->trace_id = trace_id ? trace_id : 1;
    r->span_id = 0;
    return p->n++;
}

uint64_t wubu_promote_gate(wubu_promote_t *p, int idx,
                           const char *verifier_id, int attested) {
    if (!p || !verifier_id) return (uint64_t)-1;
    if (idx < 0 || idx >= p->n) return (uint64_t)-1;
    wubu_promote_rec_t *r = &p->recs[idx];
    if (r->status != WUBU_PROMOTE_PENDING) return 0;   /* already gated */

    /* INDEPENDENCE: verifier must differ from proposer (no self-approval). */
    int independent = (strcmp(verifier_id, r->proposer_id) != 0);
    if (!independent || !attested) {
        r->status = WUBU_PROMOTE_REJECTED;
        r->verifier_attested = 0;
        p->n_rejected++;
        return 0;   /* fail-closed: rejected, no span */
    }
    /* accepted: emit a trace span */
    r->status = WUBU_PROMOTE_ACCEPTED;
    r->verifier_attested = 1;
    strncpy(r->verifier_id, verifier_id, WUBU_PROMOTE_IDLEN - 1);
    r->span_id = p->next_span_id++;
    p->n_accepted++;
    return r->span_id;
}

const wubu_promote_rec_t *wubu_promote_get(const wubu_promote_t *p, int idx) {
    if (!p || idx < 0 || idx >= p->n) return NULL;
    return &p->recs[idx];
}

int wubu_promote_accepted(const wubu_promote_t *p) { return p ? p->n_accepted : 0; }
int wubu_promote_rejected(const wubu_promote_t *p) { return p ? p->n_rejected : 0; }
