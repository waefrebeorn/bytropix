/*
 * bench_gemm.c -- baseline + new-kernel GEMM speed/accuracy benchmark.
 *
 * Times the CURRENT scalar path (wubu_matmul_f32_scalar, the engine's de-facto
 * kernels for F32/BF16/BF16) vs the NEW tiled AVX2/AVX512-FMA kernel
 * (wubu_gemm_f32), on realistic LLM dims. Also checks numerical parity
 * (max abs err + cosine) so "improvement" is provably not a regression.
 *
 * Build: gcc -O3 -march=native -mfma -I include -o bench_gemm tools/bench_gemm.c
 *        src/wubu_gemm.c -lm -fopenmp
 */
#include "wubu_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

/* Reference scalar GEMM: C[M,N] += A[M,K] * B[K,N] (row-major). This is the
 * engine's CURRENT de-facto kernel (quantized_matmul's F32 path is equivalent
 * scalar dot). Used as the baseline + the accuracy oracle. */
static void ref_gemm(const float *A, const float *B, float *C,
                     int M, int K, int N) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        const float *ar = A + (size_t)i * K;
        float *cr = C + (size_t)i * N;
        for (int j = 0; j < N; j++) cr[j] = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = ar[k];
            const float *br = B + (size_t)k * N;
            for (int j = 0; j < N; j++) cr[j] += a * br[j];
        }
    }
}

static float randf(void) { return ((float)rand() / RAND_MAX - 0.5f) * 0.1f; }

typedef struct { int M, K, N; const char *name; } dim_t;

int main(void) {
    srand(1234);
    dim_t dims[] = {
        {512, 2048, 2048,  "KAT  proj  d_model x d_ff"},
        {2048, 2048, 2048, "KAT  gate/up (d_model x d_ff)"},
        {2048, 512, 2048,  "KAT  down (d_ff x d_model)"},
        {5120, 15360, 5120,"Qwen proj  d_model x d_ff"},
        {2560, 2560, 6912, "Agents proj"},
        {4096, 11008, 4096,"Llama-7B style proj"},
    };
    int nd = sizeof(dims) / sizeof(dims[0]);

    printf("# GEMM kernel benchmark  (GFLOP/s = 2*M*K*N / sec)\n");
    printf("%-28s %8s %10s %12s %12s %10s %10s\n",
           "dims (MxKxN)", "GFLOP", "ref_ms", "new_ms", "new_GFLOPs",
           "speedup", "maxErr");

    for (int d = 0; d < nd; d++) {
        int M = dims[d].M, K = dims[d].K, N = dims[d].N;
        size_t aSz = (size_t)M*K, bSz = (size_t)K*N, cSz = (size_t)M*N;
        float *A = malloc(aSz*4), *B = malloc(bSz*4),
              *Cr = malloc(cSz*4), *Cn = malloc(cSz*4);
        for (size_t i = 0; i < aSz; i++) A[i] = randf();
        for (size_t i = 0; i < bSz; i++) B[i] = randf();

        double t0 = now_ms();
        ref_gemm(A, B, Cr, M, K, N);
        double t1 = now_ms();
        double ref_ms = t1 - t0;

        t0 = now_ms();
        wubu_gemm_f32(A, B, Cn, M, K, N);
        t1 = now_ms();
        double new_ms = t1 - t0;

        double gflop = 2.0 * M * K * N / 1e9;
        double speedup = ref_ms / new_ms;

        /* accuracy vs reference */
        float maxerr = 0.0f, nrm_r = 0, nrm_n = 0, dot = 0;
        for (size_t i = 0; i < cSz; i++) {
            float e = fabsf(Cn[i] - Cr[i]);
            if (e > maxerr) maxerr = e;
            nrm_r += Cr[i]*Cr[i]; nrm_n += Cn[i]*Cn[i]; dot += Cr[i]*Cn[i];
        }
        float cos = dot / (sqrtf(nrm_r)*sqrtf(nrm_n) + 1e-12f);

        printf("%-28s %8.1f %10.2f %12.2f %12.1f %9.2fx %10.2e (cos=%.5f)\n",
               dims[d].name, gflop, ref_ms, new_ms, gflop/(new_ms/1e3),
               speedup, maxerr, cos);

        free(A); free(B); free(Cr); free(Cn);
    }
    return 0;
}
