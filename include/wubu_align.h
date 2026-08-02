/*
 * wubu_align.h -- Preference alignment + unlearning (Theme IM). C11.
 *
 * Convergence (Rafailov et al 2023 DPO; Ethayarajh et al 2024 KTO;
 * gradient-ascent + KL-anchored unlearning, Eldan & Russinovich 2023,
 * TOFU Maini et al 2024):
 *   - IM01 DPO: the preference pair's implicit reward
 *         r = beta * (log pi(y_w|x)/pi_ref(y_w|x) -
 *                     log pi(y_l|x)/pi_ref(y_l|x));
 *         the loss = -log sigmoid(r) -- reward-model-free alignment.
 *   - IM02 KTO: binary desirability -- one outcome per example,
 *         the Kahneman-Tversky reference-point utility.
 *   - IM03 Gradient-ascent unlearning: forget-set loss ascent
 *         (the approximate exact-unlearning).
 *   - IM04 KL-anchored unlearning: forget via ascent PLUS a KL anchor
 *         to the original model -- forget without the collapse.
 *   - IM05 Alignment replay: preference-ranked reservoir replay.
 *   - IM06 Drift monitor: reward-hacking / value-drift detection
 *         (the alignment health check).
 *   - IM07 The operator: preference-satisfying config promotion on the
 *         (alignment, cost) frontier.
 */
#ifndef WUBU_ALIGN_H
#define WUBU_ALIGN_H

#include <stdint.h>

/* IM01: the DPO implicit reward for a (win, lose) pair.
 * log_pi_w/l = the model's log-probs; log_ref_w/l = the reference's;
 * beta = the temperature. Returns the reward. */
float wubu_dpo_reward(float log_pi_w, float log_ref_w,
                      float log_pi_l, float log_ref_l, float beta);
/* The DPO loss (-log sigmoid(reward)). */
float wubu_dpo_loss(float log_pi_w, float log_ref_w,
                    float log_pi_l, float log_ref_l, float beta);

/* IM02: the KTO loss for ONE example.
 * desirable = 1 for the chosen, 0 for the rejected; r = the implicit
 * reward; z_ref = the reference point; lambda_w/l = the per-side
 * weights. */
float wubu_kto_loss(int desirable, float r, float z_ref,
                    float lambda_w, float lambda_l, float beta);

/* IM03/IM04: unlearning updates.
 * The gradient-ascent update: theta -= lr * grad_forget.
 * The KL-anchored variant returns the weight on the anchor term:
 * alpha_anchor = alpha (the caller adds alpha * KL(theta || theta0)
 * to the objective). Both are pure math; the caller owns the model. */
float wubu_unlearn_ascent(float lr, float grad_forget);
float wubu_unlearn_anchor_weight(float alpha, float kl_theta_theta0);

/* IM05: the alignment replay buffer (preference-ranked reservoir). */
#define WUBU_ALIGN_BUFSZ 256
typedef struct {
    float pref[WUBU_ALIGN_BUFSZ];     /* preference score per sample */
    uint8_t used[WUBU_ALIGN_BUFSZ];
    int head, count;
    float min_pref;                   /* the current minimum kept */
} wubu_align_buffer_t;

int   wubu_align_push(wubu_align_buffer_t *b, float preference);
/* Sample the top-k preference entries (the alignment replay batch). */
int   wubu_align_topk(const wubu_align_buffer_t *b, int k, int *out_idx);
float wubu_align_mean(const wubu_align_buffer_t *b);

/* IM06: the drift monitor -- reward-hacking / value-drift detection.
 * The windowed mean/std vs the baseline; drifted when the mean moves
 * more than drift_sigma * std or the std collapses (reward hacking
 * often shows as a spiked mean + collapsed variance). */
typedef struct {
    double sum, sum2;
    int    n;
    float  baseline_mean, baseline_std;
    float  drift_sigma;
} wubu_align_monitor_t;

int   wubu_align_monitor_init(wubu_align_monitor_t *m, float drift_sigma);
int   wubu_align_monitor_feed(wubu_align_monitor_t *m, float reward);
int   wubu_align_monitor_drifted(const wubu_align_monitor_t *m);

/* IM07: the operator -- pick the config on the (alignment, cost)
 * frontier: alignment in [0,1] (higher better), cost >= 0. Returns the
 * index of the config with the max alignment among those under
 * max_cost (ties: lower cost). */
int wubu_align_pick_config(const float *alignment, const float *cost,
                           int n, float max_cost);

#endif
