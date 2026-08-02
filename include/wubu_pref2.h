/*
 * wubu_pref2.h -- the preference-alignment frontier, complete (IQ).
 * C11, agnostic + data-driven: one aligner with a loss-selector table,
 * configurable gates/schedules/telemetry, so the caller picks the
 * mechanism instead of the module hardcoding it. Covers CPO/RE-PO/
 * AlphaPO-style shaping, multi-objective + three-way, conflict
 * detection, provenance, curriculum, transfer, joint align+unlearn,
 * reward-hack pre-detection, and the operator.
 */
#ifndef WUBU_PREF2_H
#define WUBU_PREF2_H

#include <stdint.h>

/* The loss-selector table (agnostic: the caller picks). */
enum {
    WUBU_PREF_L_DPO = 0, WUBU_PREF_L_SIMPO, WUBU_PREF_L_IPO,
    WUBU_PREF_L_CPO, WUBU_PREF_L_REPO, WUBU_PREF_L_MARGIN
};

typedef struct {
    int    loss;           /* the selector */
    float  beta, gamma, tau;
    float  margin;         /* the margin schedule value */
    float  eps;            /* the label-noise robustness */
    float  kl_weight;      /* the forget-anchor tie (IM04) */
    int    use_length_norm;
} wubu_align_cfg_t;

/* IQ02: CPO -- conditional preference on discriminative prompts. */
float wubu_pref2_cpo(float logp_win, float logp_lose, float cond_score,
                     float beta);

/* IQ21: three-way multi-objective loss (win/lose/tie). */
float wubu_pref2_threeway(float logp_win, float logp_lose, int tie,
                          float beta);

/* IQ23: reward-free calibration check (reference-free alignment). */
float wubu_pref2_calib(float win_rate, float expected);

/* IQ24: conflict detection -- contradictory pair scoring. */
int wubu_pref2_conflict(const float *a, const float *b, int n, float tol);

/* IQ25: RE-PO robustness envelope (the loss is bounded). */
float wubu_pref2_envelope(float loss, float radius);

/* IQ26: budget allocation -- which prompts deserve pairs. */
int wubu_pref2_alloc(float prompt_value, float budget, float *spent);

/* IQ27: alignment-without-forgetting (preference + KL anchor). */
float wubu_pref2_anchor(float pref_loss, float kl, float w);

/* IQ29: implicit reward trace per token. */
int wubu_pref2_reward_trace(const float *logp, int n, float *trace);

/* IQ30: benchmark harness -- the aggregate alignment score. */
float wubu_pref2_bench(const float *wins, const float *loses, int n);

/* IQ32: reward-model distillation into the implicit reward. */
float wubu_pref2_distill(float implicit, float rm_score, float w);

/* IQ33: synthetic pair augmentation (the rejected-sample reuse). */
int wubu_pref2_augment(const float *rejected, int n, float *pair_out);

/* IQ34: drift monitor during fine-tune. */
int wubu_pref2_drift(float ref_win, float cur_win, float th);

/* IQ35: pair curriculum -- the domain order. */
int wubu_pref2_curriculum(float difficulty, float progress, float *w);

/* IQ36: AlphaPO-style reward shaping. */
float wubu_pref2_shape(float reward, float scale, float bias);

/* IQ37: mini-batch preference mixing. */
float wubu_pref2_batch_mix(float new_grad, float old_grad, float alpha);

/* IQ38: hard-pair emphasis. */
float wubu_pref2_hard_weight(float gap, float th);

/* IQ39: preference-regularized decode (constrained sampling). */
int wubu_pref2_constrained(const float *logits, int n, int pref_id,
                           float bias, float *out);

/* IQ40: alignment energy accounting. */
float wubu_pref2_energy(long pairs, float pj_per_pair);

/* IQ41: pair provenance tagging. */
int wubu_pref2_provenance(const char *src, int *tag);

/* IQ42: multi-turn preference (conversation-level). */
float wubu_pref2_multiturn(const float *turn_rewards, int n);

/* IQ43: staleness decay (the pair age weight). */
float wubu_pref2_stale_weight(float age, float half_life);

/* IQ44: quality gate (reject low-agreement pairs). */
int wubu_pref2_quality(float agreement, float th);

/* IQ45: DPO-vs-RLHF divergence metric. */
float wubu_pref2_method_div(float dpo_loss, float rlhf_loss);

/* IQ46: preference ensemble (multiple reward hypotheses). */
float wubu_pref2_ensemble(const float *rewards, int n, const float *w);

/* IQ47: the alignment health dashboard composite. */
float wubu_pref2_health(float acc, float drift, float margin);

/* IQ49: online bootstrap (self-generated pairs). */
int wubu_pref2_bootstrap(float self_conf, float th);

/* IQ51: length-robust reward normalization (SimPO's answer). */
float wubu_pref2_len_robust(float logp, int len, float alpha);

/* IQ52: confidence-scaled sampling temperature. */
float wubu_pref2_conf_temp(float confidence);

/* IQ53: pair-margin prediction. */
float wubu_pref2_margin_predict(const float *feat, int n);

/* IQ55: the alignment verification gate. */
int wubu_pref2_verify_gate(float eval_score, float th);

/* IQ56: transfer from a small aligned model. */
float wubu_pref2_transfer(float small_align, float sim);

/* IQ57: reward-hack pre-detection. */
int wubu_pref2_hack_detect(float reward, float expected, float dev);

/* IQ58: active pair selection (fewer, better pairs). */
int wubu_pref2_active(float uncertainty, float budget);

/* IQ59: preference entropy (pair-distribution flatness). */
float wubu_pref2_entropy(const float *probs, int n);

/* IQ60: joint align + unlearn objective. */
float wubu_pref2_joint(float pref_loss, float forget_loss, float w);

/* IQ61: align-then-select model pick. */
int wubu_pref2_select(const float *evals, int n);

/* IQ64: margin regularization (avoid over-confidence). */
float wubu_pref2_margin_reg(float margin, float cap);

/* IQ66: test-time alignment scaling. */
float wubu_pref2_tts(float logit, float budget_left);

/* IQ67: the preference-to-policy operator. */
int wubu_pref2_operator(float health, float th, int *promoted);

#endif
