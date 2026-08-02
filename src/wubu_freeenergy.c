/*
 * wubu_freeenergy.c -- Predictive coding / free energy / active
 * inference (Theme IN). C11, deterministic.
 */
#include "wubu_freeenergy.h"
#include <math.h>

float wubu_fe_pred_error(float x, float mu_hat)
{
    return x - mu_hat;
}

float wubu_fe_precision_weight(float error, float precision)
{
    if (precision < 0) precision = 0;
    return precision * error;
}

float wubu_fe_free_energy(float log_likelihood, float complexity)
{
    /* F = -accuracy + complexity; the accuracy = log p(o|s) <= 0 */
    if (complexity < 0) complexity = 0;
    float f = -log_likelihood + complexity;
    if (f < 0) f = 0;
    return f;
}

float wubu_fe_expected_free_energy(float pragmatic, float epistemic)
{
    /* pragmatic is -E[log p(o|s)]: MORE negative = better goal fit.
     * G = pragmatic + epistemic; the caller picks min G. */
    return pragmatic + epistemic;
}

int wubu_fe_policy_prior(const float *G, int n, float gamma, float *out)
{
    if (!G || !out || n <= 0) return -1;
    if (gamma < 0) gamma = 0;
    float m = G[0];
    for (int i = 1; i < n; i++) if (G[i] > m) m = G[i];
    float sum = 0;
    for (int i = 0; i < n; i++) {
        out[i] = expf(-gamma * (G[i] - m));
        sum += out[i];
    }
    for (int i = 0; i < n; i++) out[i] /= sum;
    return 0;
}

float wubu_fe_percept_step(float mu_hat, float error, float precision, float lr)
{
    if (lr < 0) lr = 0;
    return mu_hat + lr * wubu_fe_precision_weight(error, precision);
}

float wubu_fe_epistemic_value(float uncertainty_before, float uncertainty_after)
{
    if (uncertainty_before < 0) uncertainty_before = 0;
    if (uncertainty_after < 0) uncertainty_after = 0;
    if (uncertainty_after >= uncertainty_before) return 0;
    return uncertainty_before - uncertainty_after;
}

int wubu_fe_pick_model(const float *fe, const float *complexity,
                       int n, float max_complexity)
{
    if (!fe || !complexity || n <= 0) return -1;
    int best = -1;
    for (int i = 0; i < n; i++) {
        if (complexity[i] > max_complexity) continue;
        if (best < 0 || fe[i] < fe[best]) best = i;
    }
    return best;
}
