/*
 * wubu_bridge.c -- Cross-resource bridges (Theme JF, first batch). C11.
 */
#include "wubu_bridge.h"
#include <math.h>

int wubu_br_mood_retrieve(const float *mood_patterns, int n_moods,
                          float valence, float *out_mood)
{
    if (!mood_patterns || n_moods <= 0 || !out_mood) return -1;
    int best = 0;
    float bd = fabsf(mood_patterns[0] - valence);
    for (int i = 1; i < n_moods; i++) {
        float d = fabsf(mood_patterns[i] - valence);
        if (d < bd) { bd = d; best = i; }
    }
    *out_mood = mood_patterns[best];
    return best;
}

int wubu_br_confidence_gate(float confidence, float thresh, int budget_left)
{
    if (confidence < 0) confidence = 0;
    if (confidence > 1) confidence = 1;
    return (confidence >= thresh && budget_left > 0) ? 1 : 0;
}

int wubu_br_persona_guard(float safety_score, float bar)
{
    return safety_score >= bar ? 1 : 0;
}

float wubu_br_credit(float calibration, float outcome)
{
    if (calibration < 0) calibration = 0;
    if (calibration > 1) calibration = 1;
    return calibration * outcome;
}

int wubu_br_chat_prune(int turns, int budget)
{
    if (turns <= 0) return 0;
    if (budget <= 0) return turns;
    /* keep the sink (first) turn + budget-1 others */
    int keep = budget < turns ? budget : turns;
    return turns - keep;
}

int wubu_br_regulate(float confidence, int budget_left, float low, float mid)
{
    if (budget_left <= 0) return 0;             /* no budget -> stop */
    if (confidence >= mid) return 2;            /* confident -> delegate */
    if (confidence >= low) return 1;            /* uncertain -> retry */
    return 0;                                   /* low -> stop */
}

float wubu_br_mood_predict(float current, int dt, float kernel_scale)
{
    if (kernel_scale <= 0) kernel_scale = 1;
    if (dt < 0) dt = 0;
    /* exponential decay kernel: the mood relaxes toward neutral */
    return current * expf(-(float)dt / kernel_scale);
}

float wubu_br_forget_retain(int age, float halflife)
{
    if (age <= 0) return 1.0f;
    if (halflife <= 0) return 0;
    return expf(-((float)age / halflife) * 0.6931471805599453f);
}

float wubu_br_memory_weight(float base, int age, float halflife)
{
    return base * wubu_br_forget_retain(age, halflife);
}

int wubu_br_self_pattern(float capability, float *slot)
{
    if (!slot) return -1;
    if (capability < 0) capability = 0;
    if (capability > 1) capability = 1;
    *slot = capability;
    return 0;
}

int wubu_br_verify_output(float verifier_score, float bar)
{
    return verifier_score >= bar ? 1 : 0;
}

int wubu_br_monitor_log(int *log_len, int max_len)
{
    if (!log_len) return -1;
    if (*log_len < max_len) (*log_len)++;
    return *log_len;
}

int wubu_br_mood_anomaly(float mood_delta, float thresh)
{
    if (thresh < 0) thresh = 0;
    return fabsf(mood_delta) > thresh ? 1 : 0;
}

float wubu_br_empathy_reward(float emp_w, float emp_l, float beta)
{
    if (beta <= 0) beta = 1;
    return beta * (emp_w - emp_l);
}

int wubu_br_tier(float competence, float low, float high)
{
    if (competence < low) return 0;
    if (competence < high) return 1;
    return 2;
}

int wubu_br_monitor_component(float *component, int n, const float *vals)
{
    if (!component || !vals || n <= 0) return -1;
    for (int i = 0; i < n; i++) component[i] = vals[i];
    return n;
}

float wubu_br_close_rate(float prev_rate, int closed_this_batch, int batch_size)
{
    if (batch_size <= 0) return prev_rate;
    float frac = (float)closed_this_batch / (float)batch_size;
    if (prev_rate <= 0) return frac;
    return 0.7f * prev_rate + 0.3f * frac;   /* EMA */
}
