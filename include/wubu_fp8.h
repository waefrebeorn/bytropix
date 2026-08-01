/*
 * wubu_fp8.h — FP8 E4M3 / E5M2 emulation (doc B07, CPU path).
 *
 * NVIDIA/AMD GPUs have native FP8 (E4M3 for fwd, E5M2 for grad). On CPU we
 * emulate: pack F32<->FP8 with the IEEE-754-like E4M3/E5M2 formats, and provide
 * a dequantized dot product (FP8 weights x F32 activations -> F32) so the same
 * kernel shape used on GPU runs on CPU. This is the pure-CPU analog of the
 * "FP8 mixed precision" gap — no third-party lib, no GPU required.
 */
#ifndef WUBU_FP8_H
#define WUBU_FP8_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* E4M3: 1 sign, 4 exp (bias 7), 3 mantissa. Max normal ~448, NaN/inf represented
 * as the all-ones exponent (S111 1111 = NaN/Inf per microsoft spec; we treat
 * 1111 as Inf for simplicity and S1111111/S0000000 as the special 256/-256? No:
 * E4M3 has no Inf; 1111 111 = NaN, 1111 110 = 448 max. We map 1111xxxx -> NaN,
 * and saturate to +/-448. E5M2: 1 sign, 5 exp (bias 15), 2 mantissa; has Inf. */
uint8_t wubu_fp8_e4m3_from_f32(float x);
uint8_t wubu_fp8_e5m2_from_f32(float x);
float   wubu_fp8_e4m3_to_f32(uint8_t b);
float   wubu_fp8_e5m2_to_f32(uint8_t b);

/* Quantize a float vector [0,n) to FP8 (choice of format). out must hold n
 * bytes. Returns number of elements written. */
int wubu_fp8_quantize(const float *x, uint8_t *out, int n, int e5m2);

/* Dequantize FP8 vector [0,n) back to F32. */
void wubu_fp8_dequantize(const uint8_t *q, float *out, int n, int e5m2);

/* FP8 weight (length n) dot F32 activation (length n) -> F32 result.
 * Used for the matmul hot path on CPU where weights are FP8-packed. */
float wubu_fp8_dot(const uint8_t *w_fp8, const float *act, int n, int e5m2);

/* Batched: W is [rows*n] FP8 weights, A is [n] F32 activation; out[rows] = dot.
 * Mirrors the GPU FP8 GEMV shape so it can be swapped 1:1. */
void wubu_fp8_gemv(const uint8_t *W, const float *A, float *out,
                   int rows, int n, int e5m2);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_FP8_H */
