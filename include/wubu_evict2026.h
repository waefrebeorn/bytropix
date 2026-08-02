/*
 * wubu_evict2026.h -- KV-eviction frontier mechanisms (Theme IO, the
 * mechanisms wubu_kv_evict does NOT cover). C11.
 *
 * Convergence (2603.20397 KV survey; KeyDiff 2504.15364; KVQuant):
 *   - SnapKV observation-window pooling (1D pooled importance)
 *   - Proxy-token one-shot batch eviction (softmax-probability discard)
 *   - InfiniPot novelty distillation (novelty-weighted retention)
 *   - HASHEVICT LSH pre-attention eviction (SimHash hamming distance)
 *   - RocketKV two-stage coarse eviction + dynamic sparse selection
 *   - Ada-KV head-adaptive budget (eviction-loss upper bound)
 *   - KeyDiff key-similarity eviction + per-head sink discovery
 *   - Semantic sponsorship (semantic importance retention)
 *   - Eviction-loss model (bounded-error budget)
 *   - Block-drift guard (eviction-error compounding across blocks)
 *   - Sink-token FP16 reservation + outlier sparse store
 *   - Evict-or-compress hybrid budget
 *   - Running-softmax streaming aggregator
 *   - Threshold hysteresis + head-disparity monitor
 */
#ifndef WUBU_EVICT2026_H
#define WUBU_EVICT2026_H

#include <stdint.h>

/* SnapKV-style 1D pooling of attention scores: out[j] = max of each
 * pooling window of width w. Returns the pooled count. */
int wubu_ev_pool_obs(const float *attn, int n, int w, float *out);

/* Proxy-token one-shot eviction: score = softmax-probability mass of
 * each token against the proxy (first) token; the lowest-mass tokens
 * are dropped in one batch. keep = how many to retain; out_keep
 * receives the retained indices (descending score). */
int wubu_ev_proxy_evict(const float *scores, int n, int keep, int *out_keep);

/* InfiniPot novelty: novelty of token i = distance from the retained
 * set (min distance to the kept prototypes). */
float wubu_ev_novelty(const float *proto, int n_proto, int dim,
                      const float *vec);

/* HASHEVICT: SimHash of a vector (dim bits) + hamming distance. */
uint32_t wubu_ev_simhash(const float *v, int dim, const float *plane, int seed);
int      wubu_ev_hamming(uint32_t a, uint32_t b);

/* RocketKV two-stage: stage-1 coarse eviction by pooled scores, then
 * stage-2 keeps the top-sparse pages by query similarity. */
int wubu_ev_twostage(const float *coarse, const float *query_sim,
                     int n, int coarse_keep, int final_keep, int *out);

/* Ada-KV: head-adaptive budget. Given per-head attention dispersions
 * (higher = more spread), reallocate the total budget from sparse to
 * dispersed heads. Returns the budget for head i. */
int wubu_ev_adakv_budget(const float *dispersion, int n_heads,
                         int total_budget, int i);

/* KeyDiff: key-similarity eviction score (a token whose key is too
 * similar to a kept token is redundant). Returns 1 if redundant. */
int wubu_ev_keysim_redundant(const float *k, const float *kept,
                             int dim, float thresh);

/* Per-head sink discovery: the sink position is the index of the
 * max-attention token in the head's score vector. */
int wubu_ev_sink_pos(const float *attn, int n);

/* Semantic sponsorship: retain a token if its semantic score clears
 * the threshold (independent of the attention score). */
int wubu_ev_semantic_sponsor(float semantic, float thresh);

/* Eviction-loss model: the upper bound of the eviction loss for
 * dropping a token with accumulated attention a and total mass m:
 * loss_bound = a / m (the dropped mass fraction). */
float wubu_ev_loss_bound(float dropped_mass, float total_mass);

/* Block-drift guard: the eviction error accumulates across blocks;
 * returns the compounded drift after step with per-step drift d. */
float wubu_ev_block_drift(float drift_so_far, float step_drift, float cap);

/* Sink FP16 reservation: given a quantized budget, how many tokens may
 * stay at full precision (the sink + outliers). */
int wubu_ev_reserve_sink(int budget, int sink_count, int outlier_count);

/* Evict-or-compress hybrid: choose eviction when the token's value is
 * below the compress cost. Returns 1 = evict, 0 = compress. */
int wubu_ev_hybrid_choose(float value, float evict_cost, float compress_cost);

/* Running-softmax streaming aggregator: update the running max + sum
 * with a new logit; returns the softmax weight of the new token. */
float wubu_ev_stream_softmax(float *running_max, float *running_sum,
                             float logit);

/* Threshold hysteresis: decide keep/evict with a hysteresis band
 * around the threshold (avoid oscillation). state = previous decision
 * (0/1). */
int wubu_ev_hysteresis(float score, float thresh, float band, int state);

/* Head-disparity monitor: the ratio of the max to min head dispersion;
 * heads beyond the ratio need adaptive retention. */
float wubu_ev_head_disparity(const float *dispersion, int n_heads);

#endif
