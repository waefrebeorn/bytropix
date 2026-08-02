/*
 * wubu_policy.c -- Policy representation + baseline (GG02). C11.
 *
 * Convergence (policy gradient / baseline variance reduction 7-hop):
 *   - GG02: linear softmax policy π(a|s) = softmax(W·s + b). Baseline b(s)
 *     (exponential moving average of returns) centers the advantage to reduce
 *     gradient variance without bias. At home: the AGI-OS operator's config
 *     policy; baseline stabilizes learning across sweeps.
 */
#include "wubu_policy.h"
#include <math.h>
#include <string.h>

static double rng_f(unsigned *s) {
    *s = (*s * 1103515245U + 12345U) & 0x7fffffff;
    return (double)(*s) / (double)0x7fffffff;
}

int wubu_policy_init(wubu_policy_t *p, int n_actions, int state_dim, unsigned seed) {
    if (!p || n_actions <= 0 || n_actions > WUBU_POLICY_MAX_ACTIONS) return -1;
    if (state_dim <= 0 || state_dim > WUBU_POLICY_MAX_STATE) return -1;
    memset(p, 0, sizeof(*p));
    p->n_actions = n_actions;
    p->state_dim = state_dim;
    unsigned s = seed ? seed : 7;
    for (int a = 0; a < n_actions; a++) {
        p->b[a] = (rng_f(&s) - 0.5) * 0.1;
        for (int i = 0; i < state_dim; i++)
            p->W[a][i] = (rng_f(&s) - 0.5) * 0.1;
    }
    return 0;
}

int wubu_policy_probs(const wubu_policy_t *p, const double *state, double *probs) {
    if (!p || !state || !probs) return -1;
    double max_logit = -1e300;
    double logits[WUBU_POLICY_MAX_ACTIONS];
    for (int a = 0; a < p->n_actions; a++) {
        double l = p->b[a];
        for (int i = 0; i < p->state_dim; i++) l += p->W[a][i] * state[i];
        logits[a] = l;
        if (l > max_logit) max_logit = l;
    }
    double sum = 0.0;
    for (int a = 0; a < p->n_actions; a++) {
        probs[a] = exp(logits[a] - max_logit);  /* numerical stability */
        sum += probs[a];
    }
    for (int a = 0; a < p->n_actions; a++) probs[a] /= sum;
    return 0;
}

int wubu_policy_sample(const double *probs, int n_actions, unsigned *seed) {
    if (!probs || n_actions <= 0 || !seed) return -1;
    double r = rng_f(seed);
    double cum = 0.0;
    for (int a = 0; a < n_actions; a++) {
        cum += probs[a];
        if (r <= cum) return a;
    }
    return n_actions - 1;
}

int wubu_policy_update_baseline(wubu_policy_t *p, double return_val, double lr) {
    if (!p) return -1;
    /* Single scalar baseline (mean return across states). */
    if (p->baseline_n == 0) {
        p->baseline[0] = return_val;
    } else {
        double ema = (1.0 - lr) * p->baseline[0] + lr * return_val;
        p->baseline[0] = ema;
    }
    p->baseline_n++;
    return 0;
}
