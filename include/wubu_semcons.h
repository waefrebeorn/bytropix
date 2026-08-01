/*
 * wubu_semcons.h -- Semantic consensus (claim + verify + dispute) (DD04).
 */
#ifndef WUBU_SEMCONS_H
#define WUBU_SEMCONS_H

#define WUBU_SEMCONS_MAX_CLAIMS 64
#define WUBU_SEMCONS_MAX_EVIDENCE 8

typedef struct {
    int  proposer_id;
    char claim[128];
    int  verified_by[16];  /* agent IDs that verified */
    int  n_verified;
    int  disputed;
    char evidence[WUBU_SEMCONS_MAX_EVIDENCE][128];
    int  n_evidence;
} wubu_claim_t;

typedef struct {
    wubu_claim_t claims[WUBU_SEMCONS_MAX_CLAIMS];
    int n_claims;
} wubu_semcons_t;

int wubu_semcons_init(wubu_semcons_t *sc);
/* Propose a claim. Returns claim index or -1. */
int wubu_semcons_propose(wubu_semcons_t *sc, int proposer_id, const char *claim);
/* Verify a claim. Returns 1 if verified. */
int wubu_semcons_verify(wubu_semcons_t *sc, int claim_idx, int verifier_id, const char *evidence);
/* Dispute a claim. Returns 0 ok, -1 if invalid. */
int wubu_semcons_dispute(wubu_semcons_t *sc, int claim_idx, const char *evidence);
/* Check if claim has 2/3+1 verification. */
int wubu_semcons_majority_verified(const wubu_semcons_t *sc, int claim_idx, int n_agents);

#endif