/*
 * wubu_kernel.h — Hardware-agnostic kernel dispatch for WuBuOS AGI OS.
 *
 * Design: one C API, multiple backends. CPU portable C baseline
 * + device backends (CUDA/Metal/Vulkan/ROCm) register at init.
 * The engine dispatches through the table which auto-selects
 * the best backend at runtime based on workload characteristics.
 *
 * WASTE reference: https://github.com/sqliteai/waste
 * Adopted: kernel dispatch table pattern (waste_kernels[]),
 * compile-time #if backend guards, "all accelerators acceptable"
 * philosophy. Implemented fully self-contained in C11.
 *
 * C11 clean. No device backend code in this module — only the
 * dispatch interface and CPU baseline. Device backends register
 * their function pointers at runtime via wubu_kernel_register().
 */

#ifndef WUBU_KERNEL_H
#define WUBU_KERNEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel types — what the dispatch table dispatches */
typedef enum {
    WUBU_KERN_GEMM       = 0,  /* C = A*B  [M,K]x[K,N] -> [M,N] */
    WUBU_KERN_GEMV       = 1,  /* y = A*x    [M,K]x[K]  -> [M] */
    WUBU_KERN_ATTN       = 2,  /* softmax(QK^T/sqrt(d)) * V */
    WUBU_KERN_ROPE       = 3,  /* rotary positional embedding */
    WUBU_KERN_SOFTMAX    = 4,  /* row-wise softmax */
    WUBU_KERN_LAYER_NORM = 5,  /* RMSNorm */
    WUBU_KERN_QUANT      = 6,  /* fp32 -> int8/4/2 */
    WUBU_KERN_DEQUANT    = 7   /* int8/4/2 -> fp32 */
} wubu_kernel_type_t;

/* Backend IDs */
typedef enum {
    WUBU_BACKEND_AUTO    = 0,
    WUBU_BACKEND_SCALAR  = 1,  /* portable C reference */
    WUBU_BACKEND_CPU_SIMD= 2,  /* our hand-tuned SIMD */
    WUBU_BACKEND_CUDA    = 3,
    WUBU_BACKEND_METAL   = 4,
    WUBU_BACKEND_VULKAN  = 5,
    WUBU_BACKEND_ROCM    = 6,
    WUBU_BACKEND_BLAS    = 7
} wubu_backend_id_t;

/* Function pointer types — one per kernel */
typedef void (*wubu_gemm_fn)(const float *A, const float *B, float *C,
                                   int M, int K, int N, float beta);
typedef void (*wubu_gemv_fn)(const float *A, const float *x, float *y,
                                   int M, int K);
typedef void (*wubu_softmax_fn)(float *logits, int M, int N);
typedef void (*wubu_rmsnorm_fn)(float *x, const float *gamma,
                                       const float *beta, int M, int d, float eps);
typedef void (*wubu_quantize_fn)(const float *fp32, int8_t *q, float *scales,
                                        int M, int K, int bits);
typedef void (*wubu_dequantize_fn)(const int8_t *q, const float *scales,
                                         const float *zeros, float *fp32,
                                         int M, int K, int bits);
typedef void (*wubu_attn_fn)(const float *Q, const float *K, const float *V,
                                   float *out, int M, int N, int d, int n_heads, float scale);
typedef void (*wubu_rope_fn)(float *q, float *k, int d, int seq_len,
                                   float theta, int offset);

/* Single backend struct — registered at init time by device
 * backends (CUDA, Metal, Vulkan, ROCm, BLAS). The CPU
 * scalar baseline is always available. */
typedef struct wubu_kernel_backend {
    wubu_backend_id_t  id;
    const char         *name;

    /* NULL = not supported for this kernel type */
    wubu_gemm_fn       gemm;
    wubu_gemv_fn       gemv;
    wubu_attn_fn       attn;
    wubu_rope_fn       rope;
    wubu_softmax_fn    softmax;
    wubu_rmsnorm_fn    rmsnorm;
    wubu_quantize_fn   quantize;
    wubu_dequantize_fn dequantize;

    /* Probe: returns 1 if this backend handles the type.
     * NULL means it handles all types where function
     * pointers are non-NULL. */
    int (*supports)(wubu_kernel_type_t type);

    struct wubu_kernel_backend *next;
} wubu_kernel_backend_t;

/* ---- init/shutdown ---- */
int    wubu_kernel_init(void);
void   wubu_kernel_shutdown(void);

/* ---- backend registration ---- */
int wubu_kernel_register(wubu_backend_id_t id, const char *name,
                                    wubu_kernel_backend_t *backend);
int wubu_kernel_unregister(wubu_backend_id_t id);
int wubu_kernel_force_backend(wubu_backend_id_t id);

/* ---- dispatch query ---- */
const char *wubu_kernel_active_backend(wubu_kernel_type_t type);

/* ---- variadic dispatch (engine calls this) ---- */
int    wubu_kernel_run(wubu_kernel_type_t type, ...);

/* ---- helpers ---- */
const char *wubu_backend_name(wubu_backend_id_t id);
static inline int wubu_kernel_is_cpu(wubu_backend_id_t id) {
    return id == WUBU_BACKEND_SCALAR || id == WUBU_BACKEND_CPU_SIMD;
}
static inline int wubu_kernel_is_device(wubu_backend_id_t id) {
    return id >= WUBU_BACKEND_CUDA && id <= WUBU_BACKEND_BLAS;
}

/* ---- CPU baselines (always available, always correct) ---- */
/* These are the portable portable-reference implementations
 * used as fallback when no device backend is registered or
 * when it reports unsupported. They are correct and produce
 * results identical to a naive FP32 matmul within machine epsilon. */
void wubu_kernel_gemm_scalar(const float *A, const float *B, float *C,
                                    int M, int K, int N, float beta);
void wubu_kernel_gemv_scalar(const float *A, const float *x, float *y,
                                    int M, int K);
void wubu_kernel_softmax_scalar(float *logits, int M, int N);
void wubu_kernel_attention_scalar(const float *Q, const float *K, const float *V,
                                           float *out, int M, int N, int d, int n_heads, float scale);
void wubu_kernel_rope_scalar(float *q, float *k, int d, int seq_len,
                                    float theta, int offset);
void wubu_kernel_rmsnorm_scalar(float *x, const float *gamma,
                                       const float *beta, int M, int d, float eps);
void wubu_kernel_quantize_scalar(const float *fp32, int8_t *q,
                                        float *scales, int M, int K, int bits);
void wubu_kernel_dequantize_scalar(const int8_t *q, const float *scales,
                                          const float *zeros, float *fp32,
                                          int M, int K, int bits);

/* ---- compile-time backend feature macros ---- */
#if defined(WUBU_BACKEND_CUDA)
#  define WUBU_HAS_CUDA  1
#else
#  define WUBU_HAS_CUDA  0
#endif
#if defined(WUBU_BACKEND_METAL)
#  define WUBU_HAS_METAL  1
#else
#  define WUBU_HAS_METAL  0
#endif
#if defined(WUBU_BACKEND_VULKAN)
#  define WUBU_HAS_VULKAN 1
#else
#  define WUBU_HAS_VULKAN 0
#endif
#if defined(WUBU_BACKEND_ROCM)
#  define WUBU_HAS_ROCM   1
#else
#  define WUBU_HAS_ROCM   0
#endif
#if defined(WUBU_BACKEND_BLAS)
#  define WUBU_HAS_BLAS   1
#else
#  define WUBU_HAS_BLAS   0
#endif
#define WUBU_HAS_CPU  1  /* CPU baseline is always available */

/* ---- CPU feature detection ---- */
typedef struct {
    int has_avx2;
    int has_avx512;
    int has_fma;
    int l1d_kb;
    int l2_kb;
    int l3_kb;
    int n_cores;
    float mem_bw_gbs;
} wubu_cpu_features_t;

extern const wubu_cpu_features_t wubu_cpu_features;

int wubu_cpu_detect(wubu_cpu_features_t *cpu);
wubu_backend_id_t wubu_kernel_auto_select(wubu_kernel_type_t type);

/* ---- build flags ---- */
/* Define these at compile time to select a device backend:
 *   -DWUBU_BACKEND_CUDA    (CUDA device backend)
 *   -DWUBU_BACKEND_METAL   (Apple Metal backend)
 *   -DWUBU_BACKEND_VULKAN  (Vulkan compute backend)
 *   -DWUBU_BACKEND_ROCM    (ROCm/HIP backend)
 *   -DWUBU_BACKEND_BLAS    (BLAS backend, cublas/mkl)
 * If none are defined, the CPU scalar baseline is used. */

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KERNEL_H */