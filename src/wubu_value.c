/*
 * wubu_value.c -- Value iteration / Bellman backup (GG06). C11.
 *
 * Convergence (value iteration / dynamic programming 7-hop):
 *   - GG06: Bellman optimality: V*(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V*(s')].
 *     Iterated until convergence → optimal value + policy. At home: the config
 *     MDP (states=configs, actions=sweep-changes) solved via value iteration
 *     gives the globally optimal config policy (vs greedy DQN or local PG).
 */
#include "wubu_value.h"
#include <math.h>
#include <string.h>

int wubu_value_init(wubu_value_t *v, int n_states, int n_actions) {
    if (!v || n_states <= 0 || n_actions <= 0) return -1;
    memset(v, 0, sizeof(*v));
    v->n_states = n_states; v->n_actions = n_actions;
    v->gamma = 0.99;
    return 0;
}

int wubu_value_iterate(wubu_value_t *v) {
    if (!v) return -1;
    double newV[WUBU_VALUE_MAX_STATE];
    for (int s = 0; s < v->n_states; s++) {
        double best = -1e300;
        for (int a = 0; a < v->n_actions; a++) {
            double sum = v->R[s][a];
            for (int sp = 0; sp < v->n_states; sp++)
                sum += v->gamma * v->P[sp][s][a] * v->V[sp];
            if (sum > best) best = sum;
        }
        newV[s] = best;
    }
    memcpy(v->V, newV, sizeof(newV));
    return 0;
}

int wubu_value_policy(const wubu_value_t *v, int s) {
    if (!v || s < 0 || s >= v->n_states) return -1;
    int best_a = 0; double best = -1e300;
    for (int a = 0; a < v->n_actions; a++) {
        double sum = v->R[s][a];
        for (int sp = 0; sp < v->n_states; sp++)
            sum += v->gamma * v->P[sp][s][a] * v->V[sp];
        if (sum > best) { best = sum; best_a = a; }
    }
    return best_a;
}
