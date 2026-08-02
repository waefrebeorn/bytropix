/*
 * wubu_ppo.h -- PPO clipped surrogate objective (GG04).
 */
#ifndef WUBU_PPO_H
#define WUBU_PPO_H

#include "wubu_policy.h"

typedef struct {
    int   state_dim;
    int   n_actions;
    wubu_policy_t cur;   /* current policy */
    wubu_policy_t old;   /* behavior policy (before update) */
    double epsilon;      /* clip range, ~0.2 */
    double lr;
} wubu_ppo_t;

/* Init both policies identically. */
int  wubu_ppo_init(wubu_ppo_t *ppo, int n_actions, int state_dim, unsigned seed);
/* Copy current → old (call before collecting a batch). */
int  wubu_ppo_snapshot(wubu_ppo_t *ppo);
/* PPO clipped update for one (s, a, advantage). Returns clipped surrogate value. */
double wubu_ppo_update(wubu_ppo_t *ppo, const double *s, int a, double adv);

#endif