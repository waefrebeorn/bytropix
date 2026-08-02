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

#endif
