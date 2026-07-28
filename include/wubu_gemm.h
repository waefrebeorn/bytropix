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
