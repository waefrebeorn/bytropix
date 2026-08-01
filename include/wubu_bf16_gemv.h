/*
 * wubu_bf16_gemv.h -- AVX512-BF16 GEMV decode kernel (P09). Opaque-free.
 */
#ifndef WUBU_BF16_GEMV_H
#define WUBU_BF16_GEMV_H

/* y = W*x, W is [n_out x n_in] FP32. Uses AVX512-BF16 when available, else F32
 * reference. *used_bf16 reports which path ran. Returns n_out rows computed. */
int wubu_bf16_gemv(const float *W_f32, const float *x, float *y,
                   int n_out, int n_in, int *used_bf16);

#endif /* WUBU_BF16_GEMV_H */
