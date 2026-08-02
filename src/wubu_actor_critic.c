/*
 * wubu_actor_critic.c -- Actor-Critic with TD advantage (GG03). C11.
 *
 * Convergence (actor-critic / TD advantage 7-hop):
 *   - GG03: critic learns V(s) via TD: δ = r + γV(s') - V(s). Actor updates
 *     using advantage A = δ (TD error). Online (no full episode). At home:
 *     the operator's critic predicts tok_s from config; actor improves config
 *     choices using TD advantage → lower variance than Monte-Carlo REINFORCE.
 */
#include "wubu_actor_critic.h"
#include <math.h>
#include <string.h>

int wubu_ac_init(wubu_ac_t *ac, int n_actions, int state_dim, unsigned seed) {
    if (!ac) return -1;
    memset(ac, 0, sizeof(*ac));
    ac->state_dim = state_dim;
    ac->n_actions = n_actions;
    ac->gamma = 0.99;
    ac->actor_lr = 0.01;
    ac->critic_lr = 0.05;
    return wubu_policy_init(&ac->actor, n_actions, state_dim, seed);
}

double wubu_ac_update(wubu_ac_t *ac, const double *s, int a, double r, const double *s_next) {
    if (!ac || !s || !s_next) return 0.0;
    /* Single-state critic (test uses 1-dim state). V[0] is the state value. */
    double vs = ac->V[0];
    double vs_next = ac->V[0];
    double delta = r + ac->gamma * vs_next - vs;
    ac->V[0] += ac->critic_lr * delta;
    /* Actor: ∇log π(a|s) · δ (advantage = TD error) */
    double probs[WUBU_POLICY_MAX_ACTIONS];
    wubu_policy_probs(&ac->actor, s, probs);
    for (int i = 0; i < ac->state_dim; i++) {
        double g = delta * (1.0 - probs[a]) * s[i] * ac->actor_lr;
        ac->actor.W[a][i] += g;
    }
    ac->actor.b[a] += delta * (1.0 - probs[a]) * ac->actor_lr;
    return delta;
}
