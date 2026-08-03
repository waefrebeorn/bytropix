/*
 * wubu_hopfield3.h -- the associative-memory frontier, final (IP). C11.
 * Agnostic: a memory-state + policy table. Covers the remainder of
 * IP: attention-as-Hopfield retrieval, manifold curvature, federated
 * memory sharing, stabilization, cue quality, write/read batching,
 * outlier tolerance, ANN search, write/read asymmetry, decay vs
 * consolidation, retrieval-augmented memory, provenance, privacy,
 * load balancing, world-model updates, capacity warnings, importance
 * weighting, session coherence, momentum, and all remaining.
 */
#ifndef WUBU_HOPFIELD3_H
#define WUBU_HOPFIELD3_H

#include <stdint.h>

/* IP05: attention-as-Hopfield (softmax == memory read). */
int wubu_hop3_attention_read(const float *q, const float *kv, int d, int n, float *out);

/* IP22: manifold curvature estimation. */
float wubu_hop3_curvature(const float *patterns, int n, int d);

/* IP23: federated memory sharing (patterns + provenance). */
int wubu_hop3_federated(const float *patterns, int n, int src_id, int *merged);

/* IP32: memory stabilization (pattern anchoring). */
int wubu_hop3_stabilize(const float *pattern, int d, float strength, float *anchored);

/* IP41: cue embedding quality monitor. */
int wubu_hop3_cue_quality(const float *cue, int d, float th);

/* IP42: memory write batching. */
int wubu_hop3_write_batch(const float *patterns, int n, int d, int batch_size);

/* IP43: memory read batching. */
int wubu_hop3_read_batch(const float *patterns, int n, int d, int batch_size);

/* IP46: associative outlier tolerance. */
int wubu_hop3_outlier_tol(const float *pattern, int d, float max_noise);

/* IP49: memory ANN search. */
int wubu_hop3_ann(const float *query, const float *memory, int n, int d, float th, int *idx);

/* IP50: write/read asymmetry. */
float wubu_hop3_asymmetry(long write_cost, long read_benefit);

/* IP52: decay vs consolidation arbitration. */
int wubu_hop3_decay_arbitrate(float decay_rate, float rehearsal_rate, float th);

/* IP53: retrieval-augmented memory. */
int wubu_hop3_rag(const float *corpus, int n, int d, float *pattern);

/* IP54: memory provenance. */
int wubu_hop3_provenance(const float *pattern, int id, char *meta, int cap);

/* IP55: memory privacy (forget-set). */
int wubu_hop3_forget(const float *pattern, const int *forget_ids, int n, int target);

/* IP56: load balancing across tiers. */
int wubu_hop3_balance(const float *access_counts, int n, int *hot_tier);

/* IP57: world-model updates via associative memory. */
int wubu_hop3_world_update(const float *state, int d, const float *obs, float *next);

/* IP58: capacity warning. */
int wubu_hop3_capacity_warning(long stored, long limit);

/* IP59: pattern importance weighting. */
int wubu_hop3_weight(const float *patterns, int n, int d, float *weights);

/* IP60: session coherence (shared merge). */
int wubu_hop3_coherence(const float *patterns_a, const float *patterns_b, int n, int d, float *score);

/* IP61: momentum Hopfield update. */
int wubu_hop3_momentum(const float *current, const float *target, int d, float momentum, float *next);

/* IP62: sparse Hopfield (only k patterns stored). */
int wubu_hop3_sparse(const float *patterns, int n, int k, int *selected);

/* IP63: continuous-time Hopfield. */
float wubu_hop3_continuous(float tau, float input, float state);

/* IP64: Hopfield energy function. */
float wubu_hop3_energy(const float *state, const float *weights, int d);

/* IP65: capacity scaling. */
long wubu_hop3_scaling(int d, float capacity_factor);

/* IP66: noise robustness. */
int wubu_hop3_noise(const float *clean, const float *noisy, int d, float th);

/* IP67: pattern completion. */
int wubu_hop3_complete(const float *partial, int d, const float *memory, int n, float *completed);

#endif