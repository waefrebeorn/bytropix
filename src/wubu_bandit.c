/*
 * wubu_bandit.c -- Thompson Sampling / multi-armed bandits (FF06). C11.
 *
 * Convergence (Thompson sampling / Beta-Bernoulli 7-hop):
 *   - FF06: each "config family" (attention variant, quant scheme) is an arm.
 *     Posterior = Beta(α+successes, β+failures). Each round: sample from each
 *     posterior, pull argmax (allocates eval budget to promising families,
 *     exploring new ones with decreasing probability). At home: TS allocates
 *     the expensive sweep budget across config families more efficiently than
 *     round-robin or blind allocation.
 */
#include "wubu_bandit.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

static double rng_f(unsigned *s) {
    *s = (*s * 1103515245U + 12345U) & 0x7fffffff;
    return (double)(*s) / (double)0x7fffffff;
}
/* Gamma sampler (Marsaglia-Tsang) for Beta via Gamma ratio. */
static double sample_gamma(double shape, unsigned *s) {
    /* Simplified: for shape>=1, use sum of exponentials approximation. */
    double x = 0.0;
    int n = (int)shape;
    if (n < 1) n = 1;
    for (int i = 0; i < n; i++) x -= log(rng_f(s) + 1e-12);
    return x > 1e-12 ? x : 1e-12;
}
static double beta_sample(double a, double b, unsigned *s) {
    double ga = sample_gamma(a, s);
    double gb = sample_gamma(b, s);
    return ga / (ga + gb);
}

int wubu_bandit_init(wubu_bandit_t *b, int n_arms) {
    if (!b || n_arms < 1 || n_arms > WUBU_BANDIT_MAX_ARMS) return -1;
    memset(b, 0, sizeof(*b));
    b->n_arms = n_arms;
    for (int i = 0; i < n_arms; i++) { b->prior_alpha[i] = 1.0; b->prior_beta[i] = 1.0; }
    return 0;
}

int wubu_bandit_sample(const wubu_bandit_t *b, unsigned *seed) {
    if (!b || !seed || b->n_arms == 0) return -1;
    double best = -1; int best_arm = 0;
    for (int i = 0; i < b->n_arms; i++) {
        double a = b->prior_alpha[i] + b->rewards[i];
        double bb = b->prior_beta[i] + (b->pulls[i] - b->rewards[i]);
        double theta = beta_sample(a, bb, seed);
        if (theta > best) { best = theta; best_arm = i; }
    }
    return best_arm;
}

int wubu_bandit_update(wubu_bandit_t *b, int arm, int reward) {
    if (!b || arm < 0 || arm >= b->n_arms) return -1;
    b->pulls[arm]++;
    if (reward) b->rewards[arm]++;
    return 0;
}
