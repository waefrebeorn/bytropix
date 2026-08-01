/*
 * wubu_semcons.c -- Semantic consensus: claim + verify + dispute (DD04). C11.
 *
 * Convergence (AgentChain + semantic consensus 7-hop: Tendermint,
 * distributed semantic agreement, smart contract signalling):
 *   - DD04: agents propose "claims" (e.g. "config X → 27 tok/s"), other
 *     agents verify (submit evidence), disputes can be raised. A claim
 *     needs 2/3+1 verifications to be considered "consensus-valid."
 *     At home: each CoAgent verifies the others' sweep claims by re-running
 *     the measurement and submitting evidence.
 */
#include "wubu_semcons.h"
#include "wubu_bft.h"
#include <string.h>

int wubu_semcons_init(wubu_semcons_t *sc) {
    if (!sc) return -1;
    memset(sc, 0, sizeof(*sc));
    return 0;
}

int wubu_semcons_propose(wubu_semcons_t *sc, int proposer_id, const char *claim) {
    if (!sc || !claim || sc->n_claims >= WUBU_SEMCONS_MAX_CLAIMS) return -1;
    wubu_claim_t *c = &sc->claims[sc->n_claims];
    c->proposer_id = proposer_id;
    strncpy(c->claim, claim, 127);
    c->claim[127] = '\0';
    memset(c->verified_by, -1, sizeof(c->verified_by));
    c->n_verified = 0;
    c->disputed = 0;
    c->n_evidence = 0;
    return sc->n_claims++;
}

int wubu_semcons_verify(wubu_semcons_t *sc, int claim_idx, int verifier_id, const char *evidence) {
    if (!sc || claim_idx < 0 || claim_idx >= sc->n_claims) return -1;
    if (!evidence) return -1;
    wubu_claim_t *c = &sc->claims[claim_idx];
    /* Check not already verified by this agent */
    for (int i = 0; i < c->n_verified; i++)
        if (c->verified_by[i] == verifier_id) return -1;
    if (c->n_verified >= 16) return -1;
    c->verified_by[c->n_verified++] = verifier_id;
    if (evidence && c->n_evidence < WUBU_SEMCONS_MAX_EVIDENCE) {
        strncpy(c->evidence[c->n_evidence], evidence, 127);
        c->evidence[c->n_evidence][127] = '\0';
        c->n_evidence++;
    }
    return 1;
}

int wubu_semcons_dispute(wubu_semcons_t *sc, int claim_idx, const char *evidence) {
    if (!sc || claim_idx < 0 || claim_idx >= sc->n_claims || !evidence) return -1;
    wubu_claim_t *c = &sc->claims[claim_idx];
    c->disputed = 1;
    if (c->n_evidence < WUBU_SEMCONS_MAX_EVIDENCE) {
        strncpy(c->evidence[c->n_evidence], evidence, 127);
        c->evidence[c->n_evidence][127] = '\0';
        c->n_evidence++;
    }
    return 0;
}

int wubu_semcons_majority_verified(const wubu_semcons_t *sc, int claim_idx, int n_agents) {
    if (!sc || claim_idx < 0 || claim_idx >= sc->n_claims) return 0;
    int threshold = wubu_bft_threshold(n_agents);
    return (sc->claims[claim_idx].n_verified >= threshold) ? 1 : 0;
}
