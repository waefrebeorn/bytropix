/*
 * wubu_bft.c -- Byzantine Fault Tolerance consensus (DD01). C11.
 *
 * Convergence (BFT + multi-round voting 7-hop: 3RVAV, Tendermint,
 * Two-Fold BFT, n=3f+1 threshold):
 *   - DD01: simplified 3-round BFT voting (propose/accept/commit) with
 *     2/3+1 threshold. Tolerates f = floor((n-1)/3) Byzantine nodes.
 *     At home: n CoAgents propose configs; consensus on the best config
 *     even if up to f agents lie about their metrics.
 */
#include "wubu_bft.h"
#include <string.h>

int wubu_bft_threshold(int n_nodes) {
    if (n_nodes < 1) return 0;
    return (2 * n_nodes) / 3 + 1;  /* 2/3+1 */
}

int wubu_bft_init(wubu_bft_t *b, int n_nodes) {
    if (!b || n_nodes < 1 || n_nodes > WUBU_BFT_MAX_NODES) return -1;
    memset(b, 0, sizeof(*b));
    b->n_nodes = n_nodes;
    b->phase = BFT_PROPOSE;
    b->round = 1;
    b->decided = 0;
    for (int i = 0; i < n_nodes; i++) {
        b->votes[i].node_id = i;
        b->votes[i].vote = -1;
        b->votes[i].claim[0] = '\0';
    }
    return 0;
}

int wubu_bft_vote(wubu_bft_t *b, int node_id, int vote, const char *claim) {
    if (!b || node_id < 0 || node_id >= b->n_nodes) return -1;
    if (vote < -1 || vote > 1) return -1;
    b->votes[node_id].node_id = node_id;
    b->votes[node_id].vote = vote;
    if (claim)
        strncpy(b->votes[node_id].claim, claim, 127);
    b->votes[node_id].claim[127] = '\0';
    return 0;
}

int wubu_bft_majority(const wubu_bft_t *b, int phase) {
    (void)phase;
    if (!b || b->n_nodes == 0) return 0;
    int threshold = wubu_bft_threshold(b->n_nodes);
    int accept_count = 0, reject_count = 0;
    for (int i = 0; i < b->n_nodes; i++) {
        if (b->votes[i].vote == 1) accept_count++;
        if (b->votes[i].vote == 0) reject_count++;
    }
    if (accept_count >= threshold) return 1;
    if (reject_count >= threshold) return 0;
    return -1; /* no majority yet */
}

int wubu_bft_advance_phase(wubu_bft_t *b) {
    if (!b) return -1;
    if (b->decided) return b->phase;
    int maj = wubu_bft_majority(b, b->phase);
    if (maj >= 0) {
        b->decided = 1;
        b->decision = maj;
        return b->phase;
    }
    b->phase++;
    b->round++;
    return b->phase;
}

int wubu_bft_round(wubu_bft_t *b) {
    if (!b) return -1;
    for (int phase = BFT_PROPOSE; phase <= BFT_COMMIT; phase++) {
        int maj = wubu_bft_majority(b, b->phase);
        if (maj >= 0) {
            b->decided = 1;
            b->decision = maj;
            return 1;
        }
        b->phase++;
    }
    b->decided = 1;
    b->decision = (wubu_bft_majority(b, BFT_PROPOSE) >= 0) ? 1 : 0;
    return 1;
}
