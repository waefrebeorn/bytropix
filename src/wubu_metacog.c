/*
 * wubu_metacog.c -- AGI metacognition (Theme JD). C11.
 */
#include "wubu_metacog.h"
#include <math.h>
#include <string.h>

int wubu_mc_init(wubu_metacog_t *m, int n_agents)
{
    if (!m || n_agents <= 0 || n_agents > WUBU_MC_MAX_AGENTS) return -1;
    memset(m, 0, sizeof(*m));
    m->n_agents = n_agents;
    return 0;
}

int wubu_mc_set_competence(wubu_metacog_t *m, int agent, float comp)
{
    if (!m || agent < 0 || agent >= m->n_agents) return -1;
    if (comp < 0) comp = 0;
    if (comp > 1) comp = 1;
    m->competence[agent] = comp;
    m->trained = 1;
    return 0;
}

float wubu_mc_assess(const wubu_metacog_t *m, int agent, float claimed)
{
    if (!m || agent < 0 || agent >= m->n_agents) return 0;
    if (claimed < 0) claimed = 0;
    if (claimed > 1) claimed = 1;
    float comp = m->trained ? m->competence[agent] : 0.5f;
    /* clamp the claimed confidence to the competence envelope:
     * claimed' = comp + (claimed - comp) * envelope; the envelope is
     * 0.5 when untrained, tightening with calibration. */
    float env = m->trained ? 0.8f : 0.5f;
    float c = comp + (claimed - comp) * env;
    if (c < 0) c = 0;
    if (c > 1) c = 1;
    return c;
}

int wubu_mc_ece_init(wubu_mc_ece_t *e)
{
    if (!e) return -1;
    e->conf_sum = e->acc_sum = e->conf2_sum = 0;
    e->n = 0;
    return 0;
}

int wubu_mc_ece_feed(wubu_mc_ece_t *e, float confidence, float correct)
{
    if (!e) return -1;
    if (confidence < 0) confidence = 0;
    if (confidence > 1) confidence = 1;
    e->conf_sum += confidence;
    e->acc_sum += correct;
    e->conf2_sum += confidence * confidence;
    e->n++;
    return 0;
}

float wubu_mc_ece_score(const wubu_mc_ece_t *e)
{
    if (!e || e->n == 0) return 0;
    double mean_conf = e->conf_sum / e->n;
    double mean_acc = e->acc_sum / e->n;
    double ece = fabs(mean_conf - mean_acc);
    return (float)ece;
}

float wubu_mc_jol(float logprob_top1, float logprob_top2)
{
    /* the top-1 - top-2 log-prob gap; clamp to [0, 1] via 1 - exp(-gap) */
    float gap = logprob_top1 - logprob_top2;
    if (gap < 0) gap = 0;
    return 1.0f - expf(-gap);
}

int wubu_mc_harness(float jol, float low, float high, int retries_left)
{
    if (jol >= high) return 1;               /* confident -> continue */
    if (jol >= low && retries_left > 0) return 2;  /* uncertain -> retry */
    return 0;                                /* low JOL -> stop */
}

int wubu_mc_separated(float monitor_pred, float actor_pred, float drift)
{
    return fabsf(monitor_pred - actor_pred) > drift ? 1 : 0;
}

float wubu_mc_progress(float competence, float per_trial_gain, int k)
{
    if (k < 0) k = 0;
    float c = competence + per_trial_gain * (float)k;
    return c < 0 ? 0 : (c > 1 ? 1 : c);
}

float wubu_mc_recalibrate(float confidence, float entropy, float entropy_base)
{
    if (entropy_base <= 0) return confidence;
    if (entropy < 0) entropy = 0;
    float scale = entropy / entropy_base;
    if (scale > 1) scale = 1;
    /* uncertainty creep -> scale the confidence down */
    float c = confidence * (1.0f - 0.5f * scale);
    return c < 0 ? 0 : (c > 1 ? 1 : c);
}

float wubu_mc_faithfulness(float verbalized, float actual)
{
    float d = fabsf(verbalized - actual);
    return d < 0 ? 0 : (d > 1 ? 1 : d);
}

int wubu_mc_eval_run(const float *task_difficulties, int n,
                     float competence, float pass_thresh, int *passed)
{
    if (!task_difficulties || n <= 0) return -1;
    int ok = 0;
    for (int i = 0; i < n; i++)
        if (competence >= task_difficulties[i] * pass_thresh) ok++;
    if (passed) *passed = ok;
    return n;
}

int wubu_mc_delegate(const wubu_metacog_t *m, float difficulty)
{
    if (!m || m->n_agents <= 0) return -1;
    int best = 0;
    float bg = -1;
    for (int i = 0; i < m->n_agents; i++) {
        float g = m->competence[i] - difficulty;
        if (bg < 0 || g > bg) { bg = g; best = i; }
    }
    return best;
}

int wubu_mc_update_competence(wubu_metacog_t *m, int agent, float outcome,
                              float alpha)
{
    if (!m || agent < 0 || agent >= m->n_agents) return -1;
    if (alpha < 0) alpha = 0;
    if (alpha > 1) alpha = 1;
    float c = m->competence[agent] * (1.0f - alpha) + outcome * alpha;
    if (c < 0) c = 0;
    if (c > 1) c = 1;
    m->competence[agent] = c;
    return 0;
}

int wubu_mc_trace(uint32_t *trace, int *n, int max, uint32_t event)
{
    if (!trace || !n || max <= 0) return -1;
    if (*n < max) trace[(*n)++] = event;
    return *n;
}

int wubu_mc_second_order(float monitor_score, float baseline, float thresh)
{
    return fabsf(monitor_score - baseline) > thresh ? 1 : 0;
}

float wubu_mc_telemetry(const float *window, int n)
{
    if (!window || n <= 0) return 0;
    float s = 0;
    for (int i = 0; i < n; i++) s += window[i];
    return s / (float)n;
}

float wubu_mc_gap(float competence, float difficulty)
{
    return competence - difficulty;
}

float wubu_mc_sample_temp(float calibration, float base_temp)
{
    if (calibration < 0) calibration = 0;
    if (calibration > 1) calibration = 1;
    /* well-calibrated -> sharper (lower temp): temp = base * (2 - cal) */
    return base_temp * (2.0f - calibration);
}

float wubu_mc_drift(const wubu_mc_ece_t *e, float baseline_ece)
{
    float cur = wubu_mc_ece_score(e);
    return cur - baseline_ece;
}
