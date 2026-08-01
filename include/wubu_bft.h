/*
 * wubu_bft.h -- Byzantine Fault Tolerance consensus (DD01).
 */
#ifndef WUBU_BFT_H
#define WUBU_BFT_H

#define WUBU_BFT_MAX_NODES 16
#define WUBU_BFT_MAX_ROUNDS 3  /* propose / accept / commit */

typedef enum { BFT_PROPOSE = 0, BFT_ACCEPT = 1, BFT_COMMIT = 2 } wubu_bft_phase_t;

typedef struct {
    int  node_id;       /* agent ID */
    int  vote;          /* -1 = abstain, 0 = reject, 1 = accept */
    char claim[128];    /* the proposed config/metrics claim */
} wubu_bft_vote_t;

typedef struct {
    wubu_bft_vote_t votes[WUBU_BFT_MAX_NODES];
    int  n_nodes;
    int  phase;        /* current phase (BFT_PROPOSE/ACCEPT/COMMIT) */
    int  round;        /* current round number */
    char proposal[128];  /* current proposal */
    int  decided;      /* consensus reached */
    int  decision;     /* final decision (0=rejected, 1=accepted) */
} wubu_bft_t;

/* Threshold: 2/3+1 of n_nodes (tolerates f = floor((n-1)/3) Byzantine nodes). */
int  wubu_bft_threshold(int n_nodes);
int  wubu_bft_init(wubu_bft_t *b, int n_nodes);
/* Cast a vote in the current phase. */
int  wubu_bft_vote(wubu_bft_t *b, int node_id, int vote, const char *claim);
/* Advance to next phase. Returns current phase after advancement. */
int  wubu_bft_advance_phase(wubu_bft_t *b);
/* Check if current phase has 2/3+1 votes for the same value. */
int  wubu_bft_majority(const wubu_bft_t *b, int phase);
/* Full consensus round: returns 1 if decided. */
int  wubu_bft_round(wubu_bft_t *b);

#endif