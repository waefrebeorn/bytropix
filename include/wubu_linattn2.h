/*
 * wubu_linattn2.h -- the linear-attention frontier, complete (IU). C11.
 * Agnostic: a hybrid-config table (per-layer attention/SSM ratios +
 * per-head position schemes) + state ops. Covers delta-rule memory,
 * kernel autotune, precision, energy, recall compensation, streaming,
 * chunking, forget schedules, weight tying, normalization, decay,
 * quantization, parallelism, multiplexing, watchdogs, span, capacity.
 */
#ifndef WUBU_LINATTN2_H
#define WUBU_LINATTN2_H

#include <stdint.h>

/* IU21: delta-rule memory write. */
int wubu_la2_delta_write(float *state, int d, const float *k, const float *v,
                         float lr);

/* IU22: kernel-variant selection (the autotune). */
int wubu_la2_kernel_pick(const float *costs, int n);

/* IU23: state precision control. */
int wubu_la2_precision(float drift, float th_lo, float th_hi);

/* IU24: hybrid energy (SSM vs attention at 57K). */
float wubu_la2_energy(long ctx, float attn_j, float ssm_j);

/* IU25: per-layer attention/SSM scheduler. */
int wubu_la2_layer_sched(int layer, int n_layers, float ssm_frac);

/* IU26: state checkpoint. */
int wubu_la2_ckpt(const float *state, int d, float *buf);
int wubu_la2_restore(float *state, int d, const float *buf);

/* IU27: precise-recall gap metric. */
float wubu_la2_recall_gap(float linear_recall, float attn_recall);

/* IU30: constant-memory streaming. */
int wubu_la2_stream(const float *x, float *state, int d, float *out);

/* IU31: parallel chunk prefill. */
int wubu_la2_chunk_prefill(const float *x, int n, int d, int chunk,
                           float *state);

/* IU32: learned forget gates. */
float wubu_la2_forget(float gate, float base);

/* IU35: state normalization. */
int wubu_la2_normalize(float *state, int d);

/* IU37: per-state-update energy. */
float wubu_la2_update_energy(int d, float j_per_dim);

/* IU38: recurrent memory decay. */
int wubu_la2_decay(float *state, int d, float rate);

/* IU39: quantized state. */
int wubu_la2_quant_state(const float *state, int d, int bits, int32_t *out);

/* IU42: state-expansion ratio tuning. */
int wubu_la2_expansion(float recall, float target, float *ratio);

/* IU43: recurrent drafter (speculative decode). */
int wubu_la2_draft(const float *state, int d, float *logits);

/* IU44: chunk parallelism. */
int wubu_la2_chunk_par(int n_chunks, int cores);

/* IU45: gated state multiplexing. */
int wubu_la2_mux(const float **states, int n, int d, const float *gate,
                 float *out);

/* IU46: state-norm watchdog. */
int wubu_la2_watchdog(float norm, float th);

/* IU47: per-head position schemes. */
int wubu_la2_pos_head(int head, int n_heads, int *scheme);

/* IU50: effective receptive field. */
long wubu_la2_span(float decay, float th);

/* IU52: O(1) state size proof check. */
int wubu_la2_o1(int state_dims, int seq_len);

/* IU53: state-slot capacity. */
int wubu_la2_slot_cap(int d, int slots);

/* IU58: long-context needle test. */
int wubu_la2_needle(const float *state, int d, const float *needle,
                    float th);

/* IU59: state pruning (drop low-importance dims). */
int wubu_la2_prune(const float *importance, int d, float th, int *keep);

/* IU60: per-layer hybrid cost. */
float wubu_la2_layer_cost(int attn, int ssm, float a_j, float s_j);

#endif
