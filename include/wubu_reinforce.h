/*
 * wubu_reinforce.h -- REINFORCE policy gradient (GG01).
 */
#ifndef WUBU_REINFORCE_H
#define WUBU_REINFORCE_H

#include "wubu_policy.h"

#define WUBU_REINFORCE_MAX_T 256

typedef struct {
    int   state_dim;
    int   n_actions;
    /* Trajectory storage */
    double states[WUBU_REINFORCE_MAX_T][WUBU_POLICY_MAX_STATE];
    int   actions[WUBU_REINFORCE_MAX_T];
    double rewards[WUBU_REINFORCE_MAX_T];
    int   t;            /* current trajectory length */
    double gamma;
} wubu_reinforce_t;

/* Record a step. */
int wubu_reinforce_step(wubu_reinforce_t *r, const double *state, int action, double reward);
/* Compute discounted returns G_t for each step. */
int wubu_reinforce_returns(const wubu_reinforce_t *r, double *returns_out);
/* Update policy using REINFORCE (with optional baseline). Returns mean grad norm. */
double wubu_reinforce_update(wubu_reinforce_t *r, wubu_policy_t *p,
                             const double *returns, double baseline, double lr);

#endif