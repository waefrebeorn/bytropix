#ifndef LFM2_MATH_H
#define LFM2_MATH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Self-contained numeric primitives for the LFM2.5 forward.
 * No engine dependencies. C11 + <math.h> only. */

/* Plain F32 GEMM: y[M,N] = x[M,K] @ W.T, where W is stored [N,K] row-major
 * (PyTorch nn.Linear convention y = x @ W^T). Correct reference matmul. */
void lfm2_matmul_f32(const float *x, const float *W, int M, int K, int N, float *y);

/* RMSNorm: y = x / rms(x) * gamma ; rms = sqrt(mean(x^2)+eps). In place on x. */
void lfm2_rmsnorm(float *x, const float *gamma, int n, float eps);

/* BF16 -> F32 conversion (truncates the 16-bit mantissa, exact for representable
 * values; matches PyTorch's bf16 decomposition). */
static inline float lfm2_bf16_to_f32(uint16_t h) {
    uint32_t u = ((uint32_t)h) << 16;
    float f; memcpy(&f, &u, 4); return f;
}

#ifdef __cplusplus
}
#endif

#endif /* LFM2_MATH_H */
