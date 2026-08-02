/*
 * wubu_value.h -- Value iteration / Bellman backup (GG06).
 */
#ifndef WUBU_VALUE_H
#define WUBU_VALUE_H

#define WUBU_VALUE_MAX_STATE 64
#define WUBU_VALUE_MAX_ACTIONS 16

typedef struct {
    int n_states;
    int n_actions;
    /* Transition: P[s'][s][a] = prob of s' from (s,a). */
    double P[WUBU_VALUE_MAX_STATE][WUBU_VALUE_MAX_STATE][WUBU_VALUE_MAX_ACTIONS];
    double R[WUBU_VALUE_MAX_STATE][WUBU_VALUE_MAX_ACTIONS];
    double V[WUBU_VALUE_MAX_STATE];
    double gamma;
} wubu_value_t;

/* Init uniform transitions (must be normalized per (s,a)). */
int  wubu_value_init(wubu_value_t *v, int n_states, int n_actions);
/* One Bellman backup: V(s) = max_a [ R(s,a) + γ Σ_{s'} P(s'|s,a) V(s') ]. */
int  wubu_value_iterate(wubu_value_t *v);
/* Optimal policy from V: π(s) = argmax_a [ R(s,a) + γ Σ P V ]. */
int  wubu_value_policy(const wubu_value_t *v, int s);

#endif