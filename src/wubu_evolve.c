/*
 * wubu_evolve.c -- Self-evolution loop (AX06). C11.
 *
 * Convergence (DGM + self-improving agents 7-hop):
 *   - AX06: propose→verify→commit→regress loop.
 *     Proposes a code change, verifies via DGM gate + regression
 *     test, commits if verified, reverts + logs if not.
 */
#include "wubu_evolve.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WUBU_EVOLVE_MAX_HISTORY 256

int wubu_evolve_init(wubu_evolve_t *e) {
    if (!e) return -1;
    e->n_history = 0;
    e->n_accepted = 0;
    e->n_rejected = 0;
    return 0;
}

int wubu_evolve_propose(wubu_evolve_t *e, const char *proposal_id,
                              const char *description) {
    if (!e || !proposal_id) return -1;
    if (e->n_history >= WUBU_EVOLVE_MAX_HISTORY) return -1;
    wubu_evolve_entry_t *ent = &e->history[e->n_history++];
    snprintf(ent->proposal_id, sizeof(ent->proposal_id), "%s", proposal_id);
    ent->accepted = 0;
    ent->regression_passed = 0;
    ent->verified = 0;
    if (description)
        snprintf(ent->description, sizeof(ent->description), "%s", description);
    return 0;
}

int wubu_evolve_verify(wubu_evolve_t *e, const char *proposal_id,
                             int regression_passed, int verified) {
    if (!e || !proposal_id) return -1;
    for (int i = e->n_history - 1; i >= 0; i--) {
        if (strcmp(e->history[i].proposal_id, proposal_id) == 0) {
            e->history[i].regression_passed = regression_passed;
            e->history[i].verified = verified;
            e->history[i].accepted = (regression_passed && verified);
            if (e->history[i].accepted) e->n_accepted++;
            else e->n_rejected++;
            return e->history[i].accepted ? 1 : 0;
        }
    }
    return -1;  /* proposal not found */
}

int wubu_evolve_rollback(wubu_evolve_t *e, const char *proposal_id) {
    if (!e || !proposal_id) return -1;
    for (int i = e->n_history - 1; i >= 0; i--) {
        if (strcmp(e->history[i].proposal_id, proposal_id) == 0) {
            e->history[i].accepted = 0;
            e->n_rejected++;
            return 0;
        }
    }
    return -1;
}

int wubu_evolve_stats(const wubu_evolve_t *e, int *out_accepted,
                            int *out_rejected) {
    if (!e || !out_accepted || !out_rejected) return -1;
    *out_accepted = e->n_accepted;
    *out_rejected = e->n_rejected;
    return e->n_history;
}