/*
 * wubu_gemm.c -- WuBuOS-agnostic GEMM, OUR OWN CPU backend.
 *
 * Design: cache-blocked GEMM (three-level tiling: Mc/Kc/Nc) with
 *   - A panel packed into contiguous row-major (improves streaming + FMA
 *     throughput, removes strided B access in the inner loop);
 *   - B panel packed into column-major-of-blocks so the micro-kernel does a
 *     straight FMA chain with no gather;
 *   - micro-kernels: AVX2-FMA (8 lanes) and AVX512-FMA (16 lanes), plus a
 *     portable scalar fallback.
 *
 * This replaces the engine's scalar per-column dot (quantized_matmul F32/F16/
 * BF16 paths) which did no SIMD, no tiling, no packing. Numerically it is a
 * plain SGEMM (no mixed precision), so it matches a naive reference to ~1e-4.
 *
 * Triton is intentionally NOT used (CUDA-locked). This kernel is ISA-portable
 * C + x86 intrinsics; device backends register via wubu_gemm_register_device().
 */
#include "wubu_gemm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#  include <immintrin.h>
#if defined(__AVX2__)
#    define WUBU_HAVE_AVX2 1
#  endif
#  if defined(__AVX512F__)
#    define WUBU_HAVE_AVX512 1
#  endif
#endif

/* Fallback symbols so the (dead) AVX512 path still compiles when the ISA is
 * disabled (avoids undefined NR_AVX512 / MR_AVX512 at -O0/-march w/o avx512). */
#ifndef WUBU_HAVE_AVX512
#define NR_AVX512 16
#define MR_AVX512 4
#endif
#ifndef WUBU_HAVE_AVX2
#define NR_AVX2 8
#define MR_AVX2 4
#endif

/* ------------------------------------------------------------------ */
/* Scalar fallback (portable, also the accuracy oracle target)         */
/* ------------------------------------------------------------------ */
static void gemm_scalar(const float *A, const float *B, float *C,
                        int M, int K, int N) {
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

/* ------------------------------------------------------------------ */
/* AVX2-FMA micro-kernel: C[Mr, Nr] += A[Mr, Kc] * B[Kc, Nr]           */
/*   Mr = 4 rows, Nr = 8 cols (one AVX2 register). B packed col-major. */
/* ------------------------------------------------------------------ */
#if WUBU_HAVE_AVX2
#define MR_AVX2 4
#define NR_AVX2 8
static inline void kernel_avx2(const float *A, const float *Bp, float *C,
                               int Kc, int ldc, int nc) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    for (int k = 0; k < Kc; k++) {
        __m256 a0 = _mm256_set1_ps(A[0*Kc + k]);
        __m256 a1 = _mm256_set1_ps(A[1*Kc + k]);
        __m256 a2 = _mm256_set1_ps(A[2*Kc + k]);
        __m256 a3 = _mm256_set1_ps(A[3*Kc + k]);
        __m256 b0 = _mm256_loadu_ps(Bp + (size_t)k*nc + 0);
        __m256 b1 = _mm256_loadu_ps(Bp + (size_t)k*nc + NR_AVX2);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
    }
    float *c0 = C + 0*ldc, *c1 = C + 1*ldc, *c2 = C + 2*ldc, *c3 = C + 3*ldc;
    _mm256_storeu_ps(c0,      _mm256_add_ps(c00, _mm256_loadu_ps(c0)));
    _mm256_storeu_ps(c0 + 8,  _mm256_add_ps(c01, _mm256_loadu_ps(c0 + 8)));
    _mm256_storeu_ps(c1,      _mm256_add_ps(c10, _mm256_loadu_ps(c1)));
    _mm256_storeu_ps(c1 + 8,  _mm256_add_ps(c11, _mm256_loadu_ps(c1 + 8)));
    _mm256_storeu_ps(c2,      _mm256_add_ps(c20, _mm256_loadu_ps(c2)));
    _mm256_storeu_ps(c2 + 8,  _mm256_add_ps(c21, _mm256_loadu_ps(c2 + 8)));
    _mm256_storeu_ps(c3,      _mm256_add_ps(c30, _mm256_loadu_ps(c3)));
    _mm256_storeu_ps(c3 + 8,  _mm256_add_ps(c31, _mm256_loadu_ps(c3 + 8)));
}
#endif

/* ------------------------------------------------------------------ */
/* AVX512-FMA micro-kernel: Mr = 4, Nr = 16 (one zmm).                */
/* ------------------------------------------------------------------ */
#if WUBU_HAVE_AVX512
#define MR_AVX512 4
#define NR_AVX512 16
static inline void kernel_avx512(const float *A, const float *Bp, float *C,
                                 int Kc, int ldc, int nc) {
    __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
    for (int k = 0; k < Kc; k++) {
        __m512 a0 = _mm512_set1_ps(A[0*Kc + k]);
        __m512 a1 = _mm512_set1_ps(A[1*Kc + k]);
        __m512 a2 = _mm512_set1_ps(A[2*Kc + k]);
        __m512 a3 = _mm512_set1_ps(A[3*Kc + k]);
        __m512 b0 = _mm512_loadu_ps(Bp + (size_t)k*nc + 0);
        __m512 b1 = _mm512_loadu_ps(Bp + (size_t)k*nc + NR_AVX512);
        c00 = _mm512_fmadd_ps(a0, b0, c00);
        c01 = _mm512_fmadd_ps(a0, b1, c01);
        c10 = _mm512_fmadd_ps(a1, b0, c10);
        c11 = _mm512_fmadd_ps(a1, b1, c11);
        c20 = _mm512_fmadd_ps(a2, b0, c20);
        c21 = _mm512_fmadd_ps(a2, b1, c21);
        c30 = _mm512_fmadd_ps(a3, b0, c30);
        c31 = _mm512_fmadd_ps(a3, b1, c31);
    }
    float *c0 = C + 0*ldc, *c1 = C + 1*ldc, *c2 = C + 2*ldc, *c3 = C + 3*ldc;
    _mm512_storeu_ps(c0,      _mm512_add_ps(c00, _mm512_loadu_ps(c0)));
    _mm512_storeu_ps(c0 + 16, _mm512_add_ps(c01, _mm512_loadu_ps(c0 + 16)));
    _mm512_storeu_ps(c1,      _mm512_add_ps(c10, _mm512_loadu_ps(c1)));
    _mm512_storeu_ps(c1 + 16, _mm512_add_ps(c11, _mm512_loadu_ps(c1 + 16)));
    _mm512_storeu_ps(c2,      _mm512_add_ps(c20, _mm512_loadu_ps(c2)));
    _mm512_storeu_ps(c2 + 16, _mm512_add_ps(c21, _mm512_loadu_ps(c2 + 16)));
    _mm512_storeu_ps(c3,      _mm512_add_ps(c30, _mm512_loadu_ps(c3)));
    _mm512_storeu_ps(c3 + 16, _mm512_add_ps(c31, _mm512_loadu_ps(c3 + 16)));
}
#endif

/* ------------------------------------------------------------------ */
/* Packed blocked GEMM driver                                         */
/* ------------------------------------------------------------------ */
/* Tile sizes tuned for typical L2/L1 (skylake: L1 32KB, L2 1MB). */
#define MC 256
#define KC 256
#define NC 4096

static wubu_gemm_backend_t g_backend = WUBU_GEMM_AUTO;
static wubu_gemm_fn g_device_fn = NULL;
static const char *g_device_name = NULL;

static void blocked_gemm(const float *A, const float *B, float *C,
                         int M, int K, int N, int avx512) {
    /* B packed buffer: Bp[k*nc + col] for col in [0,nc). nc = panel width
     * (exact, NOT rounded). The micro-kernel reads Bp[k*nc + col..col+NR)
     * so the max read is bb + (kc_end-1)*nc + NR <= bb + kc_end*nc (since
     * NR <= nc). Allocate kc_end*nc per K-slice (worst case KC*NC). */
    int NR = avx512 ? NR_AVX512 : NR_AVX2;
    int MR = avx512 ? MR_AVX512 : MR_AVX2;
    int NP = 2 * NR;   /* panel width written by the micro-kernel (2 NR-blocks) */

    /* Parallelize over column tiles. Each thread gets its own pack buffers so
     * there is no cross-thread race on Bp/Ap. */
    #pragma omp parallel
    {
        size_t bpack_sz = (size_t)KC * NC;
        float *Bp = (float *)malloc(bpack_sz * sizeof(float));
        float *Ap = (float *)malloc((size_t)MC * KC * sizeof(float));
        if (!Bp || !Ap) { free(Bp); free(Ap); /* fall back: serial scalar */ }

        #pragma omp for schedule(dynamic, 1)
        for (int jc = 0; jc < N; jc += NC) {
            if (!Bp || !Ap) { /* rare alloc failure: whole-matrix scalar (serial) */
                gemm_scalar(A, B, C, M, K, N);
                continue;
            }
            int nc = (N - jc) < NC ? (N - jc) : NC;
            for (int kc = 0; kc < K; kc += KC) {
                int kc_end = (K - kc) < KC ? (K - kc) : KC;
                /* pack B panel: Bp[k*nc + col] = B[(kc+k), (jc+col)], col<nc */
                for (int k = 0; k < kc_end; k++) {
                    const float *bk = B + (size_t)(kc + k) * N + jc;
                    float *bp = Bp + (size_t)k * nc;
                    for (int col = 0; col < nc; col++) bp[col] = bk[col];
                }
                for (int ic = 0; ic < M; ic += MC) {
                    int mc = (M - ic) < MC ? (M - ic) : MC;
                    /* pack A block [ic:ic+mc) x [kc:kc+kc_end) */
                    for (int i = 0; i < mc; i++) {
                        memcpy(Ap + (size_t)i * kc_end,
                               A + (size_t)(ic + i) * K + kc,
                               (size_t)kc_end * sizeof(float));
                    }
                    /* micro-kernel over the (mc x kc_end) x (nc) tile. */
                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = (mc - ir) < MR ? (mc - ir) : MR;
                        for (int pc = 0; pc < nc; pc += NP) {
                            int nr = (nc - pc) < NP ? (nc - pc) : NP;
                            float *cblk = C + (size_t)(ic + ir) * N + jc + pc;
                            const float *ab = Ap + (size_t)ir * kc_end;
                            const float *bb = Bp;   /* full packed panel, stride nc */
                            if (avx512) {
#if WUBU_HAVE_AVX512
                                if (mr == MR_AVX512 && nr == NP) {
                                    kernel_avx512(ab, bb + pc, cblk, kc_end, N, nc);
                                    continue;
                                }
#endif
                            } else {
#if WUBU_HAVE_AVX2
                                if (mr == MR_AVX2 && nr == NP) {
                                    kernel_avx2(ab, bb + pc, cblk, kc_end, N, nc);
                                    continue;
                                }
#endif
                            }
                            /* fall back to scalar for partial edges */
                            for (int i = 0; i < mr; i++) {
                                float *cr = cblk + (size_t)i * N;
                                const float *ar = ab + (size_t)i * kc_end;
                                for (int k = 0; k < kc_end; k++) {
                                    float a = ar[k];
                                    const float *br = bb + (size_t)k * nc + pc;
                                    for (int j = 0; j < nr; j++) cr[j] += a * br[j];
                                }
                            }
                        }
                    }
                }
            }
        }
        free(Bp); free(Ap);
    }
}

/* ------------------------------------------------------------------ */
/* CPU feature detect                                                 */
/* ------------------------------------------------------------------ */
static int cpu_has_avx512(void) {
#if defined(__x86_64__) || defined(_M_X64)
    unsigned a, b, c, d;
    /* CPUID leaf 7 subleaf 0: EBX bit 16 = AVX512F */
    __asm__ __volatile__("cpuid" : "=a"(a),"=b"(b),"=c"(c),"=d"(d) : "0"(7),"2"(0));
    if ((b & (1u<<16)) == 0) return 0;
    /* leaf 1: ECX bit 28 = AVX, bit 12 = FMA */
    __asm__ __volatile__("cpuid" : "=a"(a),"=b"(b),"=c"(c),"=d"(d) : "0"(1));
    return ((c & (1u<<28)) && (c & (1u<<12))) ? 1 : 0;
#else
    return 0;
#endif
}

static void dispatch_f32(const float *A, const float *B, float *C,
                         int M, int K, int N, int avx512) {
    if (g_device_fn) { g_device_fn(A, B, C, M, K, N); return; }
    if (avx512) {
#if WUBU_HAVE_AVX512
        blocked_gemm(A, B, C, M, K, N, 1);
        return;
#endif
    }
#if WUBU_HAVE_AVX2
    blocked_gemm(A, B, C, M, K, N, 0);
#else
    gemm_scalar(A, B, C, M, K, N);
#endif
}

void wubu_gemm_f32_backend(wubu_gemm_backend_t b,
                           const float *A, const float *B, float *C,
                           int M, int K, int N) {
    /* zero C once (our kernel accumulates) */
    for (size_t i = 0; i < (size_t)M * N; i++) C[i] = 0.0f;
    switch (b) {
        case WUBU_GEMM_SCALAR:  gemm_scalar(A, B, C, M, K, N); break;
        case WUBU_GEMM_AVX2:
#if WUBU_HAVE_AVX2
            blocked_gemm(A, B, C, M, K, N, 0);
#else
            gemm_scalar(A, B, C, M, K, N);
#endif
            break;
        case WUBU_GEMM_AVX512:
#if WUBU_HAVE_AVX512
            blocked_gemm(A, B, C, M, K, N, 1);
#else
            gemm_scalar(A, B, C, M, K, N);
#endif
            break;
        case WUBU_GEMM_AUTO:
        default:
            dispatch_f32(A, B, C, M, K, N, cpu_has_avx512());
            break;
    }
}

void wubu_gemm_f32(const float *A, const float *B, float *C,
                   int M, int K, int N) {
    wubu_gemm_f32_backend(g_backend, A, B, C, M, K, N);
}

int wubu_gemm_register_device(wubu_gemm_fn fn, const char *name) {
    if (!fn) return -1;
    g_device_fn = fn;
    g_device_name = name ? name : "device";
    /* Device backend (if present) takes precedence over CPU. */
    return 0;
}

const char *wubu_gemm_active_backend(void) {
    if (g_device_fn) return g_device_name ? g_device_name : "device";
    if (g_backend == WUBU_GEMM_AVX512) return "avx512";
    if (g_backend == WUBU_GEMM_AVX2)   return "avx2";
    if (g_backend == WUBU_GEMM_SCALAR) return "scalar";
    return cpu_has_avx512() ? "avx512(auto)" : "avx2(auto)";
}
