/*
 * wubu_reinforce.c -- REINFORCE policy gradient (GG01). C11.
 *
 * Convergence (REINFORCE / policy gradient theorem 7-hop):
 *   - GG01: ∇J(θ) = E[Σ_t ∇log π(a_t|s_t) · (G_t - b)]. Monte-Carlo returns
 *     G_t = Σ_{τ≥t} γ^{τ-t} R_τ. On-policy, unbiased. At home: the AGI-OS
 *     operator learns π(config) maximizing expected tok_s via REINFORCE.
 */
#include "wubu_reinforce.h"
#include "wubu_policy.h"
#include <math.h>
#include <string.h>

int wubu_reinforce_step(wubu_reinforce_t *r, const double *state, int action, double reward) {
    if (!r || r->t >= WUBU_REINFORCE_MAX_T) return -1;
    memcpy(r->states[r->t], state, sizeof(double) * r->state_dim);
    r->actions[r->t] = action;
    r->rewards[r->t] = reward;
    r->t++;
    return 0;
}

int wubu_reinforce_returns(const wubu_reinforce_t *r, double *returns_out) {
    if (!r || !returns_out) return -1;
    /* G_t = Σ_{k>=t} γ^{k-t} R_k */
    for (int t = 0; t < r->t; t++) {
        double g = 0.0, disc = 1.0;
        for (int k = t; k < r->t; k++) {
            g += disc * r->rewards[k];
            disc *= r->gamma;
        }
        returns_out[t] = g;
    }
    return 0;
}

double wubu_reinforce_update(wubu_reinforce_t *r, wubu_policy_t *p,
                             const double *returns, double baseline, double lr) {
    if (!r || !p || !returns) return -1;
    double probs[WUBU_POLICY_MAX_ACTIONS];
    double grad_norm = 0.0;
    for (int t = 0; t < r->t; t++) {
        wubu_policy_probs(p, r->states[t], probs);
        int a = r->actions[t];
        double adv = returns[t] - baseline;
        /* Proper per-action gradient: for action a, grad_W[a][i] = (1 - π_a)·s_i
           (standard REINFORCE log-softmax gradient for linear policy) */
        for (int i = 0; i < p->state_dim; i++) {
            double g = adv * (1.0 - probs[a]) * r->states[t][i] * lr;
            p->W[a][i] += g;
            grad_norm += g * g;
        }
        p->b[a] += adv * (1.0 - probs[a]) * lr;
        grad_norm += adv * adv * (1.0 - probs[a]) * (1.0 - probs[a]);
    }
    return sqrt(grad_norm);
}
