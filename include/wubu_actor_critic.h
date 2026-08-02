/*
 * wubu_actor_critic.h -- Actor-Critic with TD advantage (GG03).
 */
#ifndef WUBU_ACTOR_CRITIC_H
#define WUBU_ACTOR_CRITIC_H

#include "wubu_policy.h"

typedef struct {
    int   state_dim;
    int   n_actions;
    double V[WUBU_POLICY_MAX_STATE];   /* critic: state value */
    wubu_policy_t actor;
    double gamma;
    double actor_lr;
    double critic_lr;
} wubu_ac_t;

/* Init: actor policy + zero critic. */
int  wubu_ac_init(wubu_ac_t *ac, int n_actions, int state_dim, unsigned seed);
/* TD update: given transition (s, a, r, s'), update critic (V) + actor (π).
   Returns TD error (advantage). */
double wubu_ac_update(wubu_ac_t *ac, const double *s, int a, double r, const double *s_next);

#endif