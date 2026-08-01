/*
 * wubu_evolve.h -- Self-evolution loop (AX06).
 */
#ifndef WUBU_EVOLVE_H
#define WUBU_EVOLVE_H

#define WUBU_EVOLVE_MAX_HISTORY 256

typedef struct {
    char proposal_id[128];
    char description[512];
    int accepted;
    int regression_passed;
    int verified;
} wubu_evolve_entry_t;

typedef struct {
    wubu_evolve_entry_t history[WUBU_EVOLVE_MAX_HISTORY];
    int n_history;
    int n_accepted;
    int n_rejected;
} wubu_evolve_t;

int wubu_evolve_init(wubu_evolve_t *e);
int wubu_evolve_propose(wubu_evolve_t *e, const char *proposal_id,
                                       const char *description);
int wubu_evolve_verify(wubu_evolve_t *e, const char *proposal_id,
                                     int regression_passed, int verified);
int wubu_evolve_rollback(wubu_evolve_t *e, const char *proposal_id);
int wubu_evolve_stats(const wubu_evolve_t *e,
                                  int *out_accepted, int *out_rejected);

#endif