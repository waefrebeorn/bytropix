/*
 * wubu_sae.h -- 256-bit binary signature matching for circuit probes.
 *
 * solidcv's mechinterp module quantises SAE atoms and residual states into
 * 256-bit signatures and finds the nearest atom by Hamming distance. The point
 * is that no FP16 matmul is needed: an argmin over popcounts replaces a dense
 * matrix product. This is that argmin, without the tensor-core kernel.
 */
#ifndef WUBU_SAE_H
#define WUBU_SAE_H

#include <stdint.h>

typedef struct {
    uint32_t id;
    uint8_t  sig[32];   /* 256 bits */
} wubu_sae_atom;

/* Nearest atom by Hamming distance. Ties go to the lowest index (deterministic).
 * Returns 0, or -1 on bad arguments. */
int wubu_sae_match(const uint8_t query[32], const wubu_sae_atom *atoms, int n,
                   int *best_index, int *best_dist);

/* Batch form with a distance ceiling; unmatched queries get index -1. */
int wubu_sae_match_batch(const uint8_t *queries, int n_queries,
                         const wubu_sae_atom *atoms, int n_atoms,
                         int max_dist, int *out_index, int *out_dist);

/* Build a signature from a float vector: a bit is set when the value is above
 * the vector's mean (sign-based quantisation, the cheap SAE-style encoding). */
void wubu_sae_signature(const float *v, int n, uint8_t sig[32]);

#endif /* WUBU_SAE_H */