/*
 * wubu_align.c -- Preference alignment + unlearning (Theme IM). C11.
 */
#include "wubu_align.h"
#include <math.h>
#include <string.h>

static float logit(float x)
{
    if (x > 0) return -logf(1.0f / (1.0f + expf(-x)));
    float e = expf(x);
    return logf(e / (1.0f + e));
}

float wubu_dpo_reward(float log_pi_w, float log_ref_w,
                      float log_pi_l, float log_ref_l, float beta)
{
    if (beta <= 0) beta = 1.0f;
    float rw = log_pi_w - log_ref_w;   /* implicit reward of the win */
    float rl = log_pi_l - log_ref_l;   /* implicit reward of the lose */
    return beta * (rw - rl);
}

float wubu_dpo_loss(float log_pi_w, float log_ref_w,
                    float log_pi_l, float log_ref_l, float beta)
{
    float r = wubu_dpo_reward(log_pi_w, log_ref_w, log_pi_l, log_ref_l, beta);
    /* -log sigmoid(r) = log(1 + exp(-r)) */
    return -logit(r);
}

float wubu_kto_loss(int desirable, float r, float z_ref,
                    float lambda_w, float lambda_l, float beta)
{
    if (beta <= 0) beta = 1.0f;
    if (lambda_w < 0) lambda_w = 0;
    if (lambda_l < 0) lambda_l = 0;
    if (desirable) {
        /* KTO desirable: lambda_w * sigmoid(beta * (z_ref - r));
         * the loss DROPS as the reward rises above the reference.
         * sigmoid(u) = 1/(1+e^-u) with u = beta*(z_ref - r) ->
         * 1/(1 + exp(beta*(r - z_ref))). */
        float z = beta * (r - z_ref);
        return lambda_w / (1.0f + expf(z));
    }
    /* KTO undesirable: lambda_l * sigmoid(beta * (r - z_ref));
     * the loss RISES as the reward rises (a rejected high-reward
     * output is the reward-hacking signature). */
    float z = beta * (z_ref - r);
    return lambda_l / (1.0f + expf(z));
}

float wubu_unlearn_ascent(float lr, float grad_forget)
{
    if (lr <= 0) return 0;
    return lr * grad_forget;   /* the caller subtracts this from theta */
}

float wubu_unlearn_anchor_weight(float alpha, float kl_theta_theta0)
{
    if (alpha < 0) alpha = 0;
    if (kl_theta_theta0 < 0) kl_theta_theta0 = 0;
    return alpha * kl_theta_theta0;   /* the anchor penalty term */
}

int wubu_align_push(wubu_align_buffer_t *b, float preference)
{
    if (!b) return -1;
    if (!b->used[b->head]) b->count++;
    b->used[b->head] = 1;
    b->pref[b->head] = preference;
    b->head = (b->head + 1) % WUBU_ALIGN_BUFSZ;
    /* recompute the min (cheap for the test-scale buffer) */
    b->min_pref = 1e30f;
    for (int i = 0; i < WUBU_ALIGN_BUFSZ; i++)
        if (b->used[i] && b->pref[i] < b->min_pref) b->min_pref = b->pref[i];
    if (b->min_pref == 1e30f) b->min_pref = 0;
    return 0;
}

int wubu_align_topk(const wubu_align_buffer_t *b, int k, int *out_idx)
{
    if (!b || !out_idx || k <= 0) return -1;
    if (k > b->count) k = b->count;
    int idx[WUBU_ALIGN_BUFSZ];
    int n = 0;
    for (int i = 0; i < WUBU_ALIGN_BUFSZ; i++)
        if (b->used[i]) idx[n++] = i;
    /* selection sort by descending preference */
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (b->pref[idx[j]] > b->pref[idx[i]]) {
                int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
            }
    for (int i = 0; i < k; i++) out_idx[i] = idx[i];
    return k;
}

float wubu_align_mean(const wubu_align_buffer_t *b)
{
    if (!b || b->count == 0) return 0;
    float s = 0;
    for (int i = 0; i < WUBU_ALIGN_BUFSZ; i++)
        if (b->used[i]) s += b->pref[i];
    return s / (float)b->count;
}

int wubu_align_monitor_init(wubu_align_monitor_t *m, float drift_sigma)
{
    if (!m || drift_sigma <= 0) return -1;
    m->sum = m->sum2 = 0;
    m->n = 0;
    m->baseline_mean = m->baseline_std = 0;
    m->drift_sigma = drift_sigma;
    return 0;
}

int wubu_align_monitor_feed(wubu_align_monitor_t *m, float reward)
{
    if (!m) return -1;
    m->sum += reward;
    m->sum2 += (double)reward * reward;
    m->n++;
    return 0;
}

static void stats(const wubu_align_monitor_t *m, float *mean, float *std)
{
    if (!m || m->n == 0) { *mean = 0; *std = 0; return; }
    double mn = m->sum / m->n;
    double var = m->sum2 / m->n - mn * mn;
    if (var < 0) var = 0;
    *mean = (float)mn;
    *std = (float)sqrt(var);
}

int wubu_align_monitor_drifted(const wubu_align_monitor_t *m)
{
    if (!m || m->n == 0) return 0;
    float mn, sd;
    stats(m, &mn, &sd);
    if (m->baseline_std > 0) {
        /* mean moved more than drift_sigma * baseline std, OR the std
         * collapsed (reward-hacking signature: spiked mean, flat var) */
        if (fabsf(mn - m->baseline_mean) > m->drift_sigma * m->baseline_std)
            return 1;
        if (m->baseline_std > 0 && sd < m->baseline_std * 0.1f)
            return 1;
        return 0;
    }
    /* first window: no baseline yet; the caller should call this only
     * after seeding the baseline via a second init or by feeding the
     * warm-up window and copying the stats. Treat the first window as
     * not drifted. */
    return 0;
}

int wubu_align_pick_config(const float *alignment, const float *cost,
                           int n, float max_cost)
{
    if (!alignment || !cost || n <= 0) return -1;
    int best = -1;
    for (int i = 0; i < n; i++) {
        if (cost[i] > max_cost) continue;
        if (best < 0 || alignment[i] > alignment[best] ||
            (alignment[i] == alignment[best] && cost[i] < cost[best]))
            best = i;
    }
    return best;
}
