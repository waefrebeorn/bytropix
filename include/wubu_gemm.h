/*
 * wubu_gemm.h -- WuBuOS-agnostic GEMM kernel front-end.
 *
 * One API, multiple backends. The CPU backend is OUR OWN hand-tuned kernel
 * (cache-blocked, AVX2-FMA / AVX512-FMA). Device backends (CUDA / Metal /
 * Vulkan / ROCm) plug in through the same wubu_gemm_f32() entry point by
 * registering a backend at startup — "all accelerators acceptable". Triton is
 * deliberately NOT used: it is CUDA-locked and contradicts the WuBuOS
 * all-accelerator mandate. Our kernel is ISA-portable C + intrinsics.
 *
 * Convention: row-major. C[M,N] = A[M,K] * B[K,N]. Caller allocates C.
 */
#ifndef WUBU_GEMM_H
#define WUBU_GEMM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Main entry. Dispatches to the best available backend for the current host
 * (runtime CPU feature detect; device backend if registered + present). */
void wubu_gemm_f32(const float *A, const float *B, float *C,
                   int M, int K, int N);

/* Force a specific CPU micro-kernel (testing / explicit control). */
typedef enum {
    WUBU_GEMM_AUTO = 0,   /* detect at runtime */
    WUBU_GEMM_SCALAR,     /* portable reference */
    WUBU_GEMM_AVX2,       /* FMA3 (most x86-64) */
    WUBU_GEMM_AVX512      /* AVX-512 F (Skylake+) */
} wubu_gemm_backend_t;

void wubu_gemm_f32_backend(wubu_gemm_backend_t b,
                           const float *A, const float *B, float *C,
                           int M, int K, int N);

/* Matrix-vector product: y[m] = sum_k A[m*K + k] * x[k], A row-major [M x K].
 * The engine's per-token decode path. Parallel over M (output rows) with
 * SIMD-FMA; uses the registered device backend if present. */
void wubu_gemv_f32(const float *A, const float *x, float *y, int M, int K);

/* Tile-aware GEMV honoring a roofline-chosen k_unroll (8 AVX2 / 16 AVX512).
 * Same math as wubu_gemv_f32, used when the tuner wants a specific unroll. */
void wubu_gemv_f32_tiled(const float *A, const float *x, float *y,
                         int M, int K, int k_unroll);

/* int8-weight GEMV: A is quantized per-row to int8 with per-row absmax scale
 * (scales[0..M-1]); accumulation is int32 then dequantized. Halves weight
 * traffic vs fp32 -- the Roofline bandwidth-bound decode lever. Caller passes
 * the pre-quantized int8 matrix (q[M*K]) and scales; if q==NULL a fresh
 * quantize is done internally (slower setup, same result). */
void wubu_gemv_i8(const int8_t *q, const float *scale,
                 const float *x, float *y, int M, int K);

/* Quantize an fp32 [M x K] matrix to int8 row-wise (absmax per row). */
void wubu_gemv_quantize_i8(const float *A, int8_t *q, float *scale, int M, int K);

/* Register a device backend. Returns 0 if accepted+available, <0 otherwise.
 * A device backend is a function with the same signature as wubu_gemm_f32.
 * (CUDA/Metal/Vulkan ports call this at init time; see wubu_kernel.h.) */
typedef void (*wubu_gemm_fn)(const float *A, const float *B, float *C,
                             int M, int K, int N);
int wubu_gemm_register_device(wubu_gemm_fn fn, const char *name);

/* Which backend is currently active (for diagnostics / reporting). */
const char *wubu_gemm_active_backend(void);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_GEMM_H */
