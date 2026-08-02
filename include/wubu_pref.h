/*
 * wubu_pref.h -- preference-optimization frontier (Theme IQ). C11.
 * SimPO/CPO/IPO/RE-PO/AlphaPO losses + the alignment infra: margins,
 * length-bias correction, pair weighting, dedup, mixing, aggregation,
 * noise robustness, token-level, caches, early stopping, staleness.
 */
#ifndef WUBU_PREF_H
#define WUBU_PREF_H

#include <stdint.h>

/* IQ01: SimPO -- reference-free, length-normalized average log-prob. */
float wubu_pref_simpo(float logp_win, float logp_lose,
                      int len_win, int len_lose, float beta, float gamma);

/* IQ03: IPO -- squared-error preference loss. */
float wubu_pref_ipo(float logp_win, float logp_lose, float beta, float tau);

/* IQ08: length-bias-corrected reward (normalized by the length^alpha). */
float wubu_pref_len_norm(float logp, int len, float alpha);

/* IQ06: margin-aware pair sampling score (prefer informative pairs). */
float wubu_pref_margin_score(float logp_win, float logp_lose, float margin);

/* IQ10: pair-difficulty weight (easy pairs down-weighted). */
float wubu_pref_difficulty_weight(float gap);

/* IQ11: reward accuracy -- the preference-vs-generation alignment. */
float wubu_pref_accuracy(const float *win_scores, const float *lose_scores,
                         int n);

/* IQ12: preference-pair dedup (near-duplicate suppression). */
int wubu_pref_dedup(const float **keys, int n, int d, const float *new_key,
                    float tol);

/* IQ13: offline/online mixing coefficient (static + live feedback). */
float wubu_pref_mix(float offline_w, int online_steps, int total);

/* IQ14: annotator consensus -> one pair (mean with disagreement flag). */
int wubu_pref_consensus(const float *votes, int n, float *out, float *spread);

/* IQ15: margin anneal (linear from start to end). */
float wubu_pref_margin_schedule(float start, float end, float t);

/* IQ16: noise-robust preference loss (sigmoid soften). */
float wubu_pref_noise_loss(float logit, float eps);

/* IQ17: token-level reward accumulation (per-token win/lose logits). */
float wubu_pref_token_reward(const float *tok_win, const float *tok_lose,
                             int n);

/* IQ19: preference gradient cache (reuse pair contributions). */
typedef struct { float key; float contrib; int valid; } wubu_pref_cache_t;
float wubu_pref_cache_get(wubu_pref_cache_t *c, float key, float fallback);
void  wubu_pref_cache_put(wubu_pref_cache_t *c, float key, float contrib);

/* IQ20: early stopping gate by reward accuracy. */
int wubu_pref_early_stop(float acc, float th, int patience, int *stale);

/* IQ22: pair staleness weight (age decay). */
float wubu_pref_staleness(float age, float half_life);

#endif
