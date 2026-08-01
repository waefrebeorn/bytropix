/*
 * wubu_nvfp4.h — NVFP4 (E2M1 + microscaling) emulation (doc B08, CPU path).
 *
 * Blackwell NVFP4 stores weights as E2M1 (1 sign, 2 exp bias 1, 1 mantissa)
 * with a per-block microscaling factor (mxfp4: 1 shared scale per 16 elements,
 * FP8 E4M3 scale). On CPU we emulate the format + the dequantized GEMV so the
 * same kernel shape used on GPU runs CPU-side. Pure C11, no 3rd-party. This is
 * the 4-bit companion to doc B07 (FP8).
 */
#ifndef WUBU_NVFP4_H
#define WUBU_NVFP4_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* E2M1: sign|exp(2,bias1)|mant(1). Values: 0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3,
 * +/-4, +/-6 (exp encodes 2^(e-1): e=0->subnorm handled as 0; e=1->*1, e=2->*2,
 * e=3->*4; mant adds 0.5*scale-unit). Max normal = 6. */
uint8_t wubu_nvfp4_from_f32(float x);
float   wubu_nvfp4_to_f32(uint8_t b);

/* MXFP4 block quantize: pack [n] floats into NVFP4 with one E4M3 scale per
 * BLOCK elements (scale = max_abs/6, clamped). out[packed] = 2 bits each.
 * scale_out has n/BLOCK entries (E4M3). Returns blocks written. */
int wubu_nvfp4_block_quantize(const float *x, uint8_t *packed,
                               uint8_t *scale_out, int n, int block);

/* MXFP4 dequant GEMV: W is block-quantized NVFP4 [rows*n], scale [rows*n/block];
 * A is F32 [n]; out[rows] = sum. Mirrors GPU NVFP4 GEMV shape. */
void wubu_nvfp4_gemv(const uint8_t *W, const uint8_t *scale, const float *A,
                      float *out, int rows, int n, int block);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_NVFP4_H */
