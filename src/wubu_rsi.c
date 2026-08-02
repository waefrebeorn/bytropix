/*
 * wubu_rsi.c -- recursive self-improvement frontier (Theme IV). C11.
 */
#include "wubu_rsi.h"
#include <math.h>
#include <string.h>

int wubu_rsi_gate(float verifier_score, float th, int *consecutive_fails)
{
    if (!consecutive_fails) return 0;
    if (verifier_score >= th) { *consecutive_fails = 0; return 1; }
    (*consecutive_fails)++;
    return 0;
}

int wubu_rsi_improve_improver(float self_score, float meta_score, float th)
{
    /* improve the improver when the meta-level is the bottleneck */
    return (meta_score < self_score && meta_score < th) ? 1 : 0;
}

int wubu_rsi_decompose(float difficulty, float budget, int depth,
                       int *n_subgoals)
{
    if (!n_subgoals || depth < 0 || difficulty <= 0) return -1;
    if (budget <= 0 || difficulty <= 1.0f) { *n_subgoals = 1; return 0; }
    int k = (int)(difficulty / 2.0f);
    if (k > 8) k = 8;
    if (k < 1) k = 1;
    *n_subgoals = k;
    return 0;
}

int wubu_rsi_prompt_mutate(const char *parent, char *child, int cap,
                           float fitness)
{
    if (!parent || !child || cap <= 0) return -1;
    int n = (int)strlen(parent);
    if (n >= cap - 1) n = cap - 2;
    int i;
    for (i = 0; i < n; i++) {
        char c = parent[i];
        /* mutate with a low probability, scaled by the fitness */
        if (fitness < 0.3f && (i % 3) == 0)
            c = (c == ' ') ? '_' : ' ';
        child[i] = c;
    }
    child[i] = 0;
    return 0;
}

float wubu_rsi_transfer(float src_perf, float similarity)
{
    /* transfer is the source performance discounted by dissimilarity */
    return src_perf * (0.5f + 0.5f * similarity);
}

float wubu_rsi_harness(float coverage, float asserts)
{
    if (coverage <= 0) return 0;
    return coverage * (0.5f + 0.5f * (asserts > 0 ? 1.0f : 0.0f));
}

int wubu_rsi_reflect(const float *win, const float *lose, int n, float *grad)
{
    if (!win || !lose || !grad || n <= 0) return -1;
    float g = 0;
    for (int i = 0; i < n; i++) g += (win[i] - lose[i]);
    *grad = g / n;
    return 0;
}

float wubu_rsi_mellowmax(const float *values, int n, float omega)
{
    if (!values || n <= 0) return 0;
    if (omega == 0) {
        float mx = values[0];
        for (int i = 1; i < n; i++) if (values[i] > mx) mx = values[i];
        return mx;
    }
    double s = 0;
    for (int i = 0; i < n; i++) s += exp(omega * values[i]);
    return log(s / n) / omega;
}

int wubu_rsi_experience(wubu_rsi_exp_t *e, int win, float value)
{
    if (!e) return -1;
    e->evals++;
    if (win) e->wins++;
    e->running = e->running * 0.95f + value * 0.05f;
    return 0;
}

int wubu_rsi_synth(float quality, float diversity, float th)
{
    /* accept self-generated data only above the quality+diversity gate */
    return (quality >= th && diversity >= 0.2f) ? 1 : 0;
}

float wubu_rsi_weak2strong(float teacher_acc, float agreement)
{
    /* the student's signal is the teacher's accuracy scaled by the
     * agreement (the weak-to-strong supervision weight) */
    return teacher_acc * agreement;
}

float wubu_rsi_scaffold(float steps_saved, float reliability)
{
    return steps_saved * reliability;
}

float wubu_rsi_awareness(float predicted, float actual)
{
    /* calibration: 1.0 = perfectly aware */
    float err = fabsf(predicted - actual);
    return err > 1.0f ? 0.0f : 1.0f - err;
}

float wubu_rsi_bounded_delta(float grad, float max_step, float budget_left)
{
    if (budget_left <= 0) return 0;
    float step = grad * (budget_left > 1.0f ? 1.0f : budget_left);
    if (step > max_step) step = max_step;
    if (step < -max_step) step = -max_step;
    return step;
}

int wubu_rsi_ft_schedule(long evals, long every, float drift)
{
    if (every <= 0) return 0;
    if (evals % every == 0) return 1;          /* the cadence */
    if (drift > 0.3f && evals % 10 == 0) return 1;  /* the drift trigger */
    return 0;
}
