/*
 * wubu_linattn.h -- linear-attention / SSM frontier (Theme IU). C11.
 * Chunkwise-parallel formulation, selective state updates, gated
 * delta / slot / RNN variants, tiled kernels, position encodings,
 * hybrid-head mixing, KV-elimination, long-context scaling, stability.
 */
#ifndef WUBU_LINATTN_H
#define WUBU_LINATTN_H

#include <stdint.h>

/* IU01: chunkwise-parallel linear attention -- the chunk recurrence.
 * state[k] = state[k-1] * decay + sum(chunk k). Returns the chunked
 * state update count. */
int wubu_la_chunk(const float *k, const float *v, const float *decay,
                  int n, int chunk, float *state, int d);

/* IU02: Mamba3-style selective state update (per-step A-B-C-SS). */
int wubu_la_selective(const float *x, const float *B, const float *C,
                      const float *A, float *state, float *out, int d);

/* IU03: Gated DeltaNet update: state += g * outer(B, (target - C'state)). */
int wubu_la_delta(const float *B, const float *target, const float *C,
                  float gate, float *state, int d);

/* IU04: Gated Slot Attention -- slots with per-slot gates. */
int wubu_la_slots(const float *x, const float **slots, int n_slots,
                  int d, float gate, float *out);

/* IU05: HGRN2 gated RNN with state expansion. */
int wubu_la_hgrn(const float *x, const float *g1, const float *g2,
                 float *state, int d, float *out);

/* IU08: tiled linear attention -- process the sequence in tiles. */
int wubu_la_tile(const float *k, const float *v, int n, int d,
                 int tile, float *state);

/* IU09: Lightning-attention-style recurrent update. */
float wubu_la_lightning(float state, float k, float v, float decay);

/* IU10: PaTH position encoding -- Householder accumulation. */
int wubu_la_householder(float *vec, int d, int steps);

/* IU11: hybrid-head attention -- attention+SSM heads per layer. */
int wubu_la_hybrid_heads(const float *x, int heads, int d,
                         int n_attn, int n_ssm, float *out);

/* IU13: SSM KV-elimination -- the recurrent state replaces the KV. */
int wubu_la_kv_free(const float *x, const float *A, float *state,
                    int d, float *out);

/* IU16: numerical-stability guard (recurrent accumulation clamp). */
float wubu_la_stable(float acc, float clamp);

#endif
