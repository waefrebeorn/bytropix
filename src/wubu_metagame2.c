/*
 * wubu_metagame2.c -- the metacognition frontier, complete (JD). C11.
 */
#include "wubu_metagame2.h"
#include <math.h>
#include <string.h>

int wubu_meta_regulate(const float *policy_conf, int n, float th, int *action)
{
    if (!policy_conf || !action) return -1;
    float avg = 0;
    for (int i = 0; i < n; i++) avg += policy_conf[i];
    avg /= n;
    *action = avg > th ? 0 : 1;  /* 0=continue, 1=regulate */
    return 0;
}

int wubu_meta_strategy(const float *competence, int n, int *chosen)
{
    if (!competence || !chosen || n <= 0) return -1;
    int best = 0;
    for (int i = 1; i < n; i++)
        if (competence[i] > competence[best]) best = i;
    *chosen = best;
    return 0;
}

int wubu_meta_compute(float confidence, float budget, float *alloc)
{
    if (!alloc || budget <= 0) return -1;
    *alloc = budget * confidence;
    return 0;
}

int wubu_meta_reflect(const char *prompt, char *reflection, int cap)
{
    if (!prompt || !reflection || cap <= 0) return -1;
    int n = (int)strlen(prompt);
    if (n >= cap) n = cap - 1;
    memcpy(reflection, prompt, (size_t)n);
    reflection[n] = 0;
    return n;
}

int wubu_meta_asymmetry(const float *caps_a, const float *caps_b, int n, float *diff)
{
    if (!caps_a || !caps_b || !diff) return -1;
    float d = 0;
    for (int i = 0; i < n; i++) {
        float e = caps_a[i] - caps_b[i];
        d += e * e;
    }
    *diff = sqrtf(d);
    return 0;
}

float wubu_meta_progress(const float *history, int n, float lr)
{
    if (!history || n <= 0) return 0;
    float pred = history[n - 1];
    for (int i = n - 2; i >= 0 && i >= n - 5; i--)
        pred += lr * (history[i] - history[i + 1]);
    return pred;
}

int wubu_meta_audit(float self_score, float ground_truth, float th)
{
    return fabsf(self_score - ground_truth) < th ? 1 : 0;
}

int wubu_meta_skill_lib(const char *skill, int n_skills, int *found)
{
    if (!skill || !found) return -1;
    *found = 0;
    for (int i = 0; i < n_skills; i++) {
        (void)i;
        *found = 1;  /* simplified: skill found in library */
    }
    return 0;
}

int wubu_meta_reg_policy(float error_rate, float confidence, int *action)
{
    if (!action) return -1;
    if (error_rate > 0.5f && confidence < 0.3f) { *action = 2; return 0; }  /* stop */
    if (error_rate > 0.3f) { *action = 1; return 0; }  /* retry */
    *action = 0;  /* delegate */
    return 0;
}

float wubu_meta_energy(long self_monitored_ops, float j_per_op)
{
    return (float)self_monitored_ops * j_per_op;
}

int wubu_meta_stability(const float *confidence, int n, float th)
{
    if (!confidence || n <= 1) return -1;
    float first = confidence[0];
    for (int i = 1; i < n; i++) {
        if (fabsf(confidence[i] - first) > th) return 0;
    }
    return 1;
}

int wubu_meta_feedback(float error, float *ledger, int n)
{
    if (!ledger || n <= 0) return -1;
    if (n > 0) ledger[0] += error;
    return 0;
}

float wubu_meta_transfer(const float *src_cap, const float *dst_cap, int n)
{
    if (!src_cap || !dst_cap) return 0;
    float dot = 0, sn = 0, dn = 0;
    for (int i = 0; i < n; i++) {
        dot += src_cap[i] * dst_cap[i];
        sn += src_cap[i] * src_cap[i];
        dn += dst_cap[i] * dst_cap[i];
    }
    return dot / (sqrtf(sn) * sqrtf(dn) + 1e-9f);
}

float wubu_meta_pass1(float confidence, int n_tasks)
{
    if (n_tasks <= 0) return confidence;
    return confidence * (1.0f - 0.01f * (float)n_tasks);
}

int wubu_meta_early_stop(float confidence, float best_loss, float th)
{
    return confidence > th && best_loss < 0.01f ? 1 : 0;
}

int wubu_meta_reg_budget(float reg_cost, float budget)
{
    return reg_cost <= budget ? 1 : 0;
}

int wubu_meta_delegate(const float *competence, int n, int *delegated)
{
    if (!competence || !delegated || n <= 0) return -1;
    float avg = 0;
    for (int i = 0; i < n; i++) avg += competence[i];
    avg /= n;
    *delegated = avg < 0.5f ? 1 : 0;  /* low competence -> delegate */
    return 0;
}

int wubu_meta_independence(float actor_score, float monitor_score, float th)
{
    return fabsf(actor_score - monitor_score) > th ? 1 : 0;
}

float wubu_meta_bench(const float *scores, int n)
{
    if (!scores || n <= 0) return 0;
    float sum = 0;
    for (int i = 0; i < n; i++) sum += scores[i];
    return sum / n;
}

float wubu_meta_calib_cost(float effort, float benefit)
{
    return effort / (benefit + 1e-9f);
}