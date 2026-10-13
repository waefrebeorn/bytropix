/* wubu_ztverify.c -- AGIEDR #301 zero-trust continuous re-verification.
 * See wubu_ztverify.h. */
#include "wubu_ztverify.h"
#include <math.h>
#include <stddef.h>

int wubu_ztverify_init(wubu_ztverify_t *z) {
    if (!z) return -1;
    for (int i = 0; i < WUBU_ZT_MAX_AGENTS; i++) {
        z->agents[i].agent_id = -1;
        z->agents[i].trusted = 0;
        z->agents[i].n_checks = 0;
        z->agents[i].n_revokes = 0;
    }
    z->n = 0;
    return 0;
}

static wubu_zt_agent_t *find(wubu_ztverify_t *z, int agent_id) {
    for (int i = 0; i < z->n; i++)
        if (z->agents[i].agent_id == agent_id) return &z->agents[i];
    return NULL;
}

int wubu_ztverify_enroll(wubu_ztverify_t *z, int agent_id, double attestation, double tolerance) {
    if (!z || z->n >= WUBU_ZT_MAX_AGENTS) return -1;
    if (tolerance < 0.0) tolerance = 0.0;
    wubu_zt_agent_t *a = &z->agents[z->n++];
    a->agent_id = agent_id;
    a->expected = attestation;
    a->tolerance = tolerance;
    a->trusted = 1;
    a->n_checks = 0;
    a->n_revokes = 0;
    return 0;
}

int wubu_ztverify_check(wubu_ztverify_t *z, int agent_id, double attestation) {
    if (!z) return 0;
    wubu_zt_agent_t *a = find(z, agent_id);
    if (!a) return 0;   /* unknown agent: fail-closed */
    a->n_checks++;
    double drift = fabs(attestation - a->expected);
    if (drift > a->tolerance) {
        a->trusted = 0;          /* revoke on drift (fail-closed) */
        a->n_revokes++;
        return 0;
    }
    /* within tolerance: re-affirm trust, update expected (continuous) */
    a->trusted = 1;
    a->expected = attestation;
    return 1;
}

int wubu_ztverify_trusted(const wubu_ztverify_t *z, int agent_id) {
    if (!z) return 0;
    for (int i = 0; i < z->n; i++)
        if (z->agents[i].agent_id == agent_id) return z->agents[i].trusted;
    return 0;   /* unknown: fail-closed */
}

int wubu_ztverify_revoke(wubu_ztverify_t *z, int agent_id) {
    if (!z) return -1;
    wubu_zt_agent_t *a = find(z, agent_id);
    if (!a) return -1;
    a->trusted = 0;
    a->n_revokes++;
    return 0;
}

uint64_t wubu_ztverify_revokes(const wubu_ztverify_t *z, int agent_id) {
    if (!z) return 0;
    for (int i = 0; i < z->n; i++)
        if (z->agents[i].agent_id == agent_id) return z->agents[i].n_revokes;
    return 0;
}
