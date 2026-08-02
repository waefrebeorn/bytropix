/*
 * wubu_pref.c -- preference-optimization frontier (Theme IQ). C11.
 */
#include "wubu_pref.h"
#include <math.h>

static float lg(float x) { return logf(1.0f + expf(-x)); }

float wubu_pref_simpo(float logp_win, float logp_lose,
                      int len_win, int len_lose, float beta, float gamma)
{
    /* BCE form: -log sigmoid(x) = log(1+e^-x) = lg(x), POSITIVE.
     * (DA: the previous -lg() sign made every loss negative.) */
    float rw = logp_win / (len_win > 0 ? len_win : 1);
    float rl = logp_lose / (len_lose > 0 ? len_lose : 1);
    return lg(beta * (rw - rl) - gamma);
}

float wubu_pref_ipo(float logp_win, float logp_lose, float beta, float tau)
{
    float diff = logp_win - logp_lose;
    float t = (diff - tau) / (2.0f * beta);
    return t * t;
}

float wubu_pref_len_norm(float logp, int len, float alpha)
{
    if (len <= 0) return logp;
    return logp / powf((float)len, alpha);
}

float wubu_pref_margin_score(float logp_win, float logp_lose, float margin)
{
    /* informative pairs sit near the margin */
    float gap = fabsf((logp_win - logp_lose) - margin);
    return 1.0f / (1.0f + gap * gap);
}

float wubu_pref_difficulty_weight(float gap)
{
    /* the weight peaks at a moderate gap, falls at extremes */
    return expf(-gap * gap);
}

float wubu_pref_accuracy(const float *win_scores, const float *lose_scores,
                         int n)
{
    if (!win_scores || !lose_scores || n <= 0) return 0;
    int ok = 0;
    for (int i = 0; i < n; i++)
        if (win_scores[i] > lose_scores[i]) ok++;
    return (float)ok / (float)n;
}

int wubu_pref_dedup(const float **keys, int n, int d, const float *new_key,
                    float tol)
{
    if (!keys || !new_key) return -1;
    for (int i = 0; i < n; i++) {
        float s = 0;
        for (int j = 0; j < d; j++) { float e = keys[i][j] - new_key[j]; s += e * e; }
        if (sqrtf(s) < tol) return i;
    }
    return -1;
}

float wubu_pref_mix(float offline_w, int online_steps, int total)
{
    if (total <= 0) return offline_w;
    /* the online fraction grows with the steps */
    float online = (float)online_steps / (float)total;
    return (1.0f - online) * offline_w + online;
}

int wubu_pref_consensus(const float *votes, int n, float *out, float *spread)
{
    if (!votes || n <= 0) return -1;
    float sum = 0;
    for (int i = 0; i < n; i++) sum += votes[i];
    float mean = sum / n;
    float var = 0;
    for (int i = 0; i < n; i++) { float e = votes[i] - mean; var += e * e; }
    var = sqrtf(var / n);
    if (out) *out = mean;
    if (spread) *spread = var;
    return var > 0.3f ? 1 : 0;   /* disagreement flag */
}

float wubu_pref_margin_schedule(float start, float end, float t)
{
    if (t <= 0) return start;
    if (t >= 1) return end;
    return start + (end - start) * t;
}

float wubu_pref_noise_loss(float logit, float eps)
{
    /* softened sigmoid: the label noise widens the loss; the BCE form
     * is POSITIVE (log(1+e^-x)) -- the same sign fix as SimPO. */
    return lg(logit * (1.0f - 2.0f * eps));
}

float wubu_pref_token_reward(const float *tok_win, const float *tok_lose,
                             int n)
{
    if (!tok_win || !tok_lose || n <= 0) return 0;
    float s = 0;
    for (int i = 0; i < n; i++) s += (tok_win[i] - tok_lose[i]);
    return s / n;
}

float wubu_pref_cache_get(wubu_pref_cache_t *c, float key, float fallback)
{
    if (!c) return fallback;
    if (c->valid && c->key == key) return c->contrib;
    return fallback;
}

void wubu_pref_cache_put(wubu_pref_cache_t *c, float key, float contrib)
{
    if (!c) return;
    c->key = key; c->contrib = contrib; c->valid = 1;
}

int wubu_pref_early_stop(float acc, float th, int patience, int *stale)
{
    if (!stale) return 0;
    if (acc >= th) { *stale = 0; return 0; }
    (*stale)++;
    return *stale >= patience ? 1 : 0;
}

float wubu_pref_staleness(float age, float half_life)
{
    if (half_life <= 0) return 0;
    return expf(-0.6931472f * age / half_life);
}
