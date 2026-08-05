/* lfm2_math.c -- self-contained numeric primitives (C11, <math.h> only).
 * SPDX-License-Identifier: WaefreBeorn-UMV3 */
#include "lfm2_math.h"
#include <math.h>

void lfm2_matmul_f32(const float *x, const float *W, int M, int K, int N, float *y) {
    for (int i = 0; i < M; i++) {
        const float *xr = x + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            /* W stored [N,K] row-major: row j = output dim j, col k = input dim k */
            const float *wr = W + (size_t)j * K;
            for (int k = 0; k < K; k++) s += xr[k] * wr[k];
            y[(size_t)i * N + j] = s;
        }
    }
}

void lfm2_rmsnorm(float *x, const float *gamma, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = sqrtf(ss / n + eps);
    float inv = (rms > 0.0f) ? 1.0f / rms : 0.0f;
    for (int i = 0; i < n; i++) x[i] = x[i] * inv * gamma[i];
}
