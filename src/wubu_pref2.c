/*
 * wubu_pref2.c -- the preference-alignment frontier, complete (IQ). C11.
 */
#include "wubu_pref2.h"
#include <math.h>
#include <string.h>

static float lg(float x) { return logf(1.0f + expf(-x)); }

float wubu_pref2_cpo(float logp_win, float logp_lose, float cond_score,
                     float beta)
{
    /* CPO: the preference loss gated by the conditional difficulty */
    return lg(beta * (logp_win - logp_lose)) * (0.5f + 0.5f * cond_score);
}

float wubu_pref2_threeway(float logp_win, float logp_lose, int tie,
                          float beta)
{
    if (tie) return 0.25f * (logp_win - logp_lose) * (logp_win - logp_lose);
    return lg(beta * (logp_win - logp_lose));
}

float wubu_pref2_calib(float win_rate, float expected)
{
    float e = fabsf(win_rate - expected);
    return e > 1.0f ? 0.0f : 1.0f - e;
}

int wubu_pref2_conflict(const float *a, const float *b, int n, float tol)
{
    if (!a || !b) return -1;
    float s = 0;
    for (int i = 0; i < n; i++) { float e = a[i] - b[i]; s += e * e; }
    return sqrtf(s) < tol ? 1 : 0;   /* near-identical = contradictory */
}

float wubu_pref2_envelope(float loss, float radius)
{
    if (loss > radius) return radius;
    if (loss < -radius) return -radius;
    return loss;
}

int wubu_pref2_alloc(float prompt_value, float budget, float *spent)
{
    if (!spent || budget <= 0) return 0;
    if (prompt_value <= 0) return 0;
    *spent += prompt_value * 0.1f;
    return *spent <= budget ? 1 : 0;
}

float wubu_pref2_anchor(float pref_loss, float kl, float w)
{
    return pref_loss + w * kl;
}

int wubu_pref2_reward_trace(const float *logp, int n, float *trace)
{
    if (!logp || !trace || n <= 0) return -1;
    float base = logp[0];
    for (int i = 0; i < n; i++) trace[i] = logp[i] - base;
    return n;
}

float wubu_pref2_bench(const float *wins, const float *loses, int n)
{
    if (!wins || !loses || n <= 0) return 0;
    float acc = 0;
    for (int i = 0; i < n; i++) if (wins[i] > loses[i]) acc += 1.0f;
    return acc / n;
}

float wubu_pref2_distill(float implicit, float rm_score, float w)
{
    return (1.0f - w) * implicit + w * rm_score;
}

int wubu_pref2_augment(const float *rejected, int n, float *pair_out)
{
    if (!rejected || !pair_out || n <= 0) return -1;
    /* the rejected sample becomes the "lose" side of a synthetic pair */
    memcpy(pair_out, rejected, sizeof(float) * n);
    return n;
}

int wubu_pref2_drift(float ref_win, float cur_win, float th)
{
    return fabsf(cur_win - ref_win) > th ? 1 : 0;
}

int wubu_pref2_curriculum(float difficulty, float progress, float *w)
{
    if (!w) return -1;
    /* easy-first: the weight peaks for the current progress band */
    float target = progress;
    float d = fabsf(difficulty - target);
    *w = expf(-d * d * 4.0f);
    return 0;
}

float wubu_pref2_shape(float reward, float scale, float bias)
{
    return reward * scale + bias;
}

float wubu_pref2_batch_mix(float new_grad, float old_grad, float alpha)
{
    return alpha * new_grad + (1.0f - alpha) * old_grad;
}

float wubu_pref2_hard_weight(float gap, float th)
{
    return gap < th ? 1.0f : (th / (gap > 0 ? gap : 1.0f));
}

int wubu_pref2_constrained(const float *logits, int n, int pref_id,
                           float bias, float *out)
{
    if (!logits || !out || n <= 0) return -1;
    for (int i = 0; i < n; i++) {
        out[i] = logits[i] + (i == pref_id ? bias : 0.0f);
    }
    return n;
}

float wubu_pref2_energy(long pairs, float pj_per_pair)
{
    return (float)pairs * pj_per_pair;
}

int wubu_pref2_provenance(const char *src, int *tag)
{
    if (!src || !tag) return -1;
    /* hash the source into a stable provenance tag */
    uint32_t h = 2166136261u;
    for (const char *p = src; *p; p++) { h ^= (unsigned char)*p; h *= 16777619u; }
    *tag = (int)(h & 0xFFFF);
    return 0;
}

float wubu_pref2_multiturn(const float *turn_rewards, int n)
{
    if (!turn_rewards || n <= 0) return 0;
    float s = 0;
    for (int i = 0; i < n; i++) s += turn_rewards[i] * expf(-0.2f * i);
    return s;
}

float wubu_pref2_stale_weight(float age, float half_life)
{
    if (half_life <= 0) return 0;
    return expf(-0.6931472f * age / half_life);
}

int wubu_pref2_quality(float agreement, float th)
{
    return agreement >= th ? 1 : 0;
}

float wubu_pref2_method_div(float dpo_loss, float rlhf_loss)
{
    return fabsf(dpo_loss - rlhf_loss);
}

float wubu_pref2_ensemble(const float *rewards, int n, const float *w)
{
    if (!rewards || n <= 0) return 0;
    float s = 0, ws = 0;
    for (int i = 0; i < n; i++) {
        float wi = w ? w[i] : 1.0f;
        s += wi * rewards[i]; ws += wi;
    }
    return ws > 0 ? s / ws : 0;
}

float wubu_pref2_health(float acc, float drift, float margin)
{
    float h = acc - drift * 0.5f + margin * 0.1f;
    return h < 0 ? 0 : (h > 1 ? 1 : h);
}

int wubu_pref2_bootstrap(float self_conf, float th)
{
    return self_conf >= th ? 1 : 0;
}

float wubu_pref2_len_robust(float logp, int len, float alpha)
{
    if (len <= 0) return logp;
    return logp / powf((float)len, alpha);
}

float wubu_pref2_conf_temp(float confidence)
{
    /* confident -> sharp (low temperature); unsure -> soft */
    return 1.0f - 0.8f * confidence;
}

float wubu_pref2_margin_predict(const float *feat, int n)
{
    if (!feat || n <= 0) return 0;
    float s = 0;
    for (int i = 0; i < n; i++) s += feat[i];
    return s / n;
}

int wubu_pref2_verify_gate(float eval_score, float th)
{
    return eval_score >= th ? 1 : 0;
}

float wubu_pref2_transfer(float small_align, float sim)
{
    return small_align * (0.5f + 0.5f * sim);
}

int wubu_pref2_hack_detect(float reward, float expected, float dev)
{
    return fabsf(reward - expected) > dev ? 1 : 0;
}

int wubu_pref2_active(float uncertainty, float budget)
{
    return (uncertainty > 0.5f && budget > 0) ? 1 : 0;
}

float wubu_pref2_entropy(const float *probs, int n)
{
    if (!probs || n <= 0) return 0;
    float h = 0;
    for (int i = 0; i < n; i++)
        if (probs[i] > 0) h -= probs[i] * logf(probs[i]);
    return h;
}

float wubu_pref2_joint(float pref_loss, float forget_loss, float w)
{
    return pref_loss + w * forget_loss;
}

int wubu_pref2_select(const float *evals, int n)
{
    if (!evals || n <= 0) return -1;
    int best = 0;
    for (int i = 1; i < n; i++) if (evals[i] > evals[best]) best = i;
    return best;
}

float wubu_pref2_margin_reg(float margin, float cap)
{
    return margin > cap ? cap : margin;
}

float wubu_pref2_tts(float logit, float budget_left)
{
    /* more budget -> more sampling freedom (scale the logit) */
    return logit * (0.5f + 0.5f * budget_left);
}

int wubu_pref2_operator(float health, float th, int *promoted)
{
    if (!promoted) return -1;
    if (health >= th) { *promoted = 1; return 1; }
    *promoted = 0;
    return 0;
}
