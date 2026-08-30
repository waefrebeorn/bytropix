/* lfm2_math.c -- self-contained numeric primitives (C11, <math.h> only).
 * SPDX-License-Identifier: WaefreBeorn-UMV3 */
#include "lfm2_math.h"
#include "gguf_reader.h"
#include <stdint.h>
#include <math.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

void lfm2_matmul_f32(const float *x, const float *W, int M, int K, int N, float *y) {
    /* Parallelize only for large enough M to avoid OpenMP overhead on small matrices */
    #pragma omp parallel for schedule(static) if(M >= 64)
    for (int i = 0; i < M; i++) {
        const float *xr = x + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            /* W stored [N,K] row-major: row j = output dim j, col k = input dim k */
            const float *wr = W + (size_t)j * K;
            int k = 0;
#ifdef __AVX512F__
            __m512 vs = _mm512_setzero_ps();
            for (; k + 15 < K; k += 16) {
                __m512 vx = _mm512_loadu_ps(xr + k);
                __m512 vw = _mm512_loadu_ps(wr + k);
                vs = _mm512_fmadd_ps(vx, vw, vs);
            }
            s = _mm512_reduce_add_ps(vs);
#elif defined(__AVX2__)
            __m256 vs = _mm256_setzero_ps();
            for (; k + 7 < K; k += 8) {
                __m256 vx = _mm256_loadu_ps(xr + k);
                __m256 vw = _mm256_loadu_ps(wr + k);
                vs = _mm256_fmadd_ps(vx, vw, vs);
            }
            __m128 hi = _mm256_extractf128_ps(vs, 1);
            __m128 lo = _mm256_castps256_ps128(vs);
            __m128 sum = _mm_add_ps(lo, hi);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            s = _mm_cvtss_f32(sum);
#endif
            for (; k < K; k++) s += xr[k] * wr[k];
            y[(size_t)i * N + j] = s;
        }
    }
}

void lfm2_qmatmul(const float *x, const float *W_f32, const uint8_t *W_q,
                  int q_type, int M, int K, int N, float *y) {
    if (W_q) {
        quantized_matmul_batched(x, W_q, q_type, K, N, 0, M, y);
        return;
    }
    lfm2_matmul_f32(x, W_f32, M, K, N, y);
}

void lfm2_rmsnorm(float *x, const float *gamma, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = sqrtf(ss / n + eps);
    float inv = (rms > 0.0f) ? 1.0f / rms : 0.0f;
    for (int i = 0; i < n; i++) x[i] = x[i] * inv * gamma[i];
}
