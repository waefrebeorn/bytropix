/* wubu_simd_physics.c — WORLDSIM #607: vectorized physics (AVX2).
 * C11, self-contained. Scalar reference + AVX2 8-wide integration over
 * SoA body arrays.
 */
#include "wubu_simd_physics.h"

void wubu_simd_integrate_scalar(const float *dx, const float *dy, const float *dz,
                                float *px, float *py, float *pz,
                                int n, float drag, float g, float dt) {
    const float keep = 1.0f - drag;
    const float gdt = g * dt;
    for (int i = 0; i < n; i++) {
        /* gravity on the Y velocity */
        float vy = dy[i] + gdt;
        /* drag on all three */
        float vx = dx[i] * keep;
        float vz = dz[i] * keep;
        vy *= keep;
        /* integrate position */
        px[i] += vx * dt;
        py[i] += vy * dt;
        pz[i] += vz * dt;
    }
}

int wubu_simd_has_avx2(void) {
#ifdef __AVX2__
    return 1;
#else
    return 0;
#endif
}

#ifdef __AVX2__
#include <immintrin.h>

void wubu_simd_integrate_avx2(const float *dx, const float *dy, const float *dz,
                              float *px, float *py, float *pz,
                              int n, float drag, float g, float dt) {
    const float keep = 1.0f - drag;
    const float gdt = g * dt;
    const __m256 vkeep = _mm256_set1_ps(keep);
    const __m256 vgdt  = _mm256_set1_ps(gdt);
    const __m256 vdt   = _mm256_set1_ps(dt);

    int i = 0;
    for (; i + 8 <= n; i += 8) {
        /* load velocities */
        __m256 vx = _mm256_loadu_ps(&dx[i]);
        __m256 vy = _mm256_loadu_ps(&dy[i]);
        __m256 vz = _mm256_loadu_ps(&dz[i]);
        /* gravity on Y */
        vy = _mm256_add_ps(vy, vgdt);
        /* drag */
        vx = _mm256_mul_ps(vx, vkeep);
        vy = _mm256_mul_ps(vy, vkeep);
        vz = _mm256_mul_ps(vz, vkeep);
        /* integrate position: pos += v*dt */
        __m256 X = _mm256_loadu_ps(&px[i]);
        __m256 Y = _mm256_loadu_ps(&py[i]);
        __m256 Z = _mm256_loadu_ps(&pz[i]);
        X = _mm256_add_ps(X, _mm256_mul_ps(vx, vdt));
        Y = _mm256_add_ps(Y, _mm256_mul_ps(vy, vdt));
        Z = _mm256_add_ps(Z, _mm256_mul_ps(vz, vdt));
        _mm256_storeu_ps(&px[i], X);
        _mm256_storeu_ps(&py[i], Y);
        _mm256_storeu_ps(&pz[i], Z);
    }
    /* scalar tail */
    for (; i < n; i++) {
        float vy = dy[i] + gdt;
        float vx = dx[i] * keep;
        float vz = dz[i] * keep;
        vy *= keep;
        px[i] += vx * dt;
        py[i] += vy * dt;
        pz[i] += vz * dt;
    }
}

#else  /* no __AVX2__: fall back to scalar (build never breaks) */

void wubu_simd_integrate_avx2(const float *dx, const float *dy, const float *dz,
                              float *px, float *py, float *pz,
                              int n, float drag, float g, float dt) {
    wubu_simd_integrate_scalar(dx, dy, dz, px, py, pz, n, drag, g, dt);
}

#endif
