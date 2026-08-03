/*
 * wubu_ternary.h -- the ternary/1.58-bit quantization frontier (JC). C11.
 * Agnostic: a quantizer-state + training schedule, the caller
 * picks the precision path. Covers 1.58-bit QAT, BitNet
 * regularizer, weight-only inference, two-phase schedule,
 * per-layer precision, activation-aware QAT, curriculum,
 * GEMV optimization, 2-bit extension, gradient handling,
 * KV quantization, transition monitor, energy accounting,
 * fine-tuning, ablation, robustness, alignment, mixed-precision,
 * evaluation harness.
 */
#ifndef WUBU_TERNARY_H
#define WUBU_TERNARY_H

#include <stdint.h>

/* JC01: 1.58-bit QAT weight quantizer. */
int wubu_ternary_qat(const float *w, int n, float alpha, int8_t *out);

/* JC02: transition schedule (when to switch precision). */
int wubu_ternary_schedule(int step, int warmup, int total, float *alpha);

/* JC03: BitNet regularizer view. */
float wubu_ternary_reg(float w_norm, float target);

/* JC04: weight-only 1.58 inference. */
int wubu_ternary_infer(const int8_t *w, int n, const float *x, float *out);

/* JC05: two-phase QAT warm-up then quantize. */
int wubu_ternary_twophase(int step, int warmup, int quant_step);

/* JC06: per-layer precision schedule. */
int wubu_ternary_layer_prec(int layer, int n_layers, float *bits);

/* JC07: activation-aware QAT. */
int wubu_ternary_act_aware(const float *act, int n, float *scale);

/* JC08: quantization curriculum. */
int wubu_ternary_curriculum(int step, int total, float *bits);

/* JC09: ternary GEMV optimization. */
int wubu_ternary_gemv(const int8_t *w, int n, const float *x, float *out);

/* JC10: 2-bit QAT extension. */
int wubu_ternary_qat_2bit(const float *w, int n, int8_t *out);

/* JC11: QAT gradient handling (straight-through). */
int wubu_ternary_grad(const float *grad, const int8_t *q, int n, float *out_grad);

/* JC12: quantization-aware KV training. */
int wubu_ternary_kv_qat(const float *kv, int n, int bits, int32_t *out);

/* JC13: precision transition monitor. */
int wubu_ternary_transition(float cur_bits, float target_bits, float th);

/* JC14: QAT energy accounting. */
float wubu_ternary_energy(long tokens, float j_per_token);

/* JC15: quantized fine-tuning. */
int wubu_ternary_finetune(const float *w, int n, int bits, int epochs);

/* JC16: bit-width ablation. */
float wubu_ternary_ablation(int bits, float baseline_acc);

/* JC17: QAT robustness. */
int wubu_ternary_robust(const float *w, int n, float noise);

/* JC18: quantization-aware alignment. */
int wubu_ternary_align(const float *w, int n, float *aligned);

/* JC19: mixed-precision QAT. */
int wubu_ternary_mixed(const float *w, int n, const int *bits, int8_t *out);

/* JC20: QAT evaluation harness. */
float wubu_ternary_eval(const float *w, const float *x, int n, int d);

#endif