/*
 * wubu_hopfield3.h -- Hopfield frontier, batch 2 (Theme IP). C11.
 * The memory-systems engineering: matrix compression, spectral
 * ranking/cleanup, dedup, condensation, chaining, energy monitor,
 * corruption detection, hygiene, fusion into attention, multi-scale,
 * snapshot/restore, capacity telemetry, beta autotuning.
 */
#ifndef WUBU_HOPFIELD3_H
#define WUBU_HOPFIELD3_H

#include <stdint.h>

/* IP15: low-rank memory matrix (store the top-k singular directions
 * via a running Gram-Schmidt: the compressed pattern bank). */
typedef struct {
    float **basis;   /* k x d */
    float  *weights; /* k energies */
    int     k, d, n;
} wubu_mem_compress_t;
int wubu_mem_compress_init(wubu_mem_compress_t *m, int k, int d);
int wubu_mem_compress_add(wubu_mem_compress_t *m, const float *pat);
int wubu_mem_compress_recall(const wubu_mem_compress_t *m, const float *cue,
                             float *out);
void wubu_mem_compress_free(wubu_mem_compress_t *m);

/* IP24: spectral overlap of a cue with a pattern bank. */
float wubu_mem_spectral_overlap(const float *cue, const float **bank,
                                int n, int d);

/* IP27: dedup -- skip a write if an identical pattern exists. */
int wubu_mem_dedup(const float **bank, int n, int d, const float *pat,
                   float tol);

/* IP28: softmax read with a temperature (sharpness per query). */
int wubu_mem_read_t(const float **bank, int n, int d, const float *cue,
                    float beta, float *out);

/* IP30: memory chaining -- given the last recall, score the next. */
float wubu_mem_chain(const float *last, const float *next, int d);

/* IP31: free-energy of the memory state (negative log-sum-exp). */
float wubu_mem_energy(const float **bank, int n, const float *cue,
                      int d, float beta);

/* IP34: corruption detection -- pattern degradation watchdog. */
int wubu_mem_corrupt(const float *pat, const float *ref, int d, float tol);

/* IP35: hygiene -- prune the stale low-utility patterns. */
int wubu_mem_prune(const float **bank, const float *utility, int n,
                   float th, int *keep, int cap);

/* IP37: attention fusion -- a retrieved pattern as a bias vector. */
int wubu_mem_attn_bias(const float *pattern, int d, float scale,
                       float *bias);

/* IP39: snapshot/restore of the pattern bank (flat serialization). */
int wubu_mem_snapshot(const float **bank, int n, int d, float *buf);
int wubu_mem_restore(float **bank, int n, int d, const float *buf);

/* IP40: capacity telemetry -- used vs theoretical (alpha*P^2). */
float wubu_mem_capacity(int n_patterns, int dim);

/* IP44: condensation -- merge near-identical patterns. */
int wubu_mem_condense(const float **bank, int n, int d, float tol,
                      float **out, int cap);

/* IP47: spectral cleanup -- drop the low-energy basis directions. */
int wubu_mem_spectral_cleanup(wubu_mem_compress_t *m, float min_energy);

/* IP51: beta autotune by recall error (gradient-ish step). */
float wubu_mem_beta_tune(float beta, float recall_err, float lr);

/* --- IP batch 3: the remaining 26 gaps (implemented in wubu_hopfield4.c) --- */

/* IP62: sparse Hopfield (only k patterns stored). */
int wubu_hop3_sparse(const float *patterns, int n, int k, int *selected);

/* IP63: continuous-time Hopfield dynamics. */
float wubu_hop3_continuous(float tau, float input, float state);

/* IP64: energy function. */
float wubu_hop3_energy(const float *state, const float *weights, int d);

/* IP65: capacity scaling. */
long wubu_hop3_scaling(int d, float capacity_factor);

/* IP66: noise robustness. */
int wubu_hop3_noise(const float *clean, const float *noisy, int d, float th);

/* IP67: pattern completion. */
int wubu_hop3_complete(const float *partial, int d, const float *memory, int n, float *completed);

/* IP05: attention-as-Hopfield retrieval (softmax == memory read). */
int wubu_hop3_attention_read(const float *q, const float *kv, int d, int n, float *out);

/* IP22: manifold curvature estimation. */
float wubu_hop3_curvature(const float *patterns, int n, int d);

/* IP23: federated memory sharing. */
int wubu_hop3_federated(const float *patterns, int n, int src_id, int *merged);

/* IP32: memory stabilization. */
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

/* IP57: world-model updates. */
int wubu_hop3_world_update(const float *state, int d, const float *obs, float *next);

/* IP58: capacity warning. */
int wubu_hop3_capacity_warning(long stored, long limit);

/* IP59: pattern importance weighting. */
int wubu_hop3_weight(const float *patterns, int n, int d, float *weights);

/* IP60: session coherence. */
int wubu_hop3_coherence(const float *a, const float *b, int n, int d, float *score);

/* IP61: momentum Hopfield update. */
int wubu_hop3_momentum(const float *current, const float *target, int d, float momentum, float *next);

#endif