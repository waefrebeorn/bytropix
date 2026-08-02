/*
 * wubu_ppo.c -- PPO clipped surrogate objective (GG04). C11.
 *
 * Convergence (PPO / clipped surrogate 7-hop):
 *   - GG04: ratio r = π_θ(a|s)/π_θ_old(a|s). L = min(r·A, clip(r,1-ε,1+ε)·A).
 *     Prevents destructive large policy updates (trust region, first-order).
 *     At home: the operator's config policy updates are clipped → stable
 *     optimization without blowing up the sweep (costs real wall-clock).
 */
#include "wubu_ppo.h"
#include <math.h>
#include <string.h>

int wubu_ppo_init(wubu_ppo_t *ppo, int n_actions, int state_dim, unsigned seed) {
    if (!ppo) return -1;
    memset(ppo, 0, sizeof(*ppo));
    ppo->epsilon = 0.2;
    ppo->lr = 0.01;
    wubu_policy_init(&ppo->cur, n_actions, state_dim, seed);
    memcpy(&ppo->old, &ppo->cur, sizeof(wubu_policy_t));
    return 0;
}

int wubu_ppo_snapshot(wubu_ppo_t *ppo) {
    if (!ppo) return -1;
    memcpy(&ppo->old, &ppo->cur, sizeof(wubu_policy_t));
    return 0;
}

double wubu_ppo_update(wubu_ppo_t *ppo, const double *s, int a, double adv) {
    if (!ppo || !s) return 0.0;
    double p_cur[WUBU_POLICY_MAX_ACTIONS], p_old[WUBU_POLICY_MAX_ACTIONS];
    wubu_policy_probs(&ppo->cur, s, p_cur);
    wubu_policy_probs(&ppo->old, s, p_old);
    double pi_cur = p_cur[a], pi_old = p_old[a];
    if (pi_old < 1e-12) pi_old = 1e-12;
    double ratio = pi_cur / pi_old;
    double clipped = (ratio > 1.0 + ppo->epsilon) ? (1.0 + ppo->epsilon)
                    : (ratio < 1.0 - ppo->epsilon) ? (1.0 - ppo->epsilon) : ratio;
    double unclipped_obj = ratio * adv;
    double clipped_obj = clipped * adv;
    double obj = (unclipped_obj < clipped_obj) ? unclipped_obj : clipped_obj;
    /* Gradient step on current policy in direction of obj (sign-based for simplicity).
       Move W[a][i] toward increasing obj: d(obj)/dW[a][i] ∝ sign(adv)·(1-π_a)·s_i·ratio. */
    double dir = (obj >= 0) ? 1.0 : -1.0;
    for (int i = 0; i < ppo->cur.state_dim; i++) {
        double g = dir * adv * (1.0 - pi_cur) * s[i] * ppo->lr;
        ppo->cur.W[a][i] += g;
    }
    ppo->cur.b[a] += dir * adv * (1.0 - pi_cur) * ppo->lr;
    return obj;
}
