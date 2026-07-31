/*
 * wubu_kernel_backends.c — Device backend registration.
 *
 * WASTE reference: "all accelerators acceptable" — device backends
 * register at runtime via wubu_kernel_register(). The engine never
 * calls a backend directly — it goes through the dispatch table which
 * auto-selects the best backend.
 *
 * Compile-time gates: WUBU_HAS_CUDA, WUBU_HAS_VULKAN, etc.
 * Runtime probes: each backend checks if the accelerator is present.
 *
 * On WSL2 GPU: CUDA runtime libs at /usr/lib/x86_64-linux-gnu,
 *               GPU passthrough at /usr/lib/wsl/lib
 * Set: LD_LIBRARY_PATH=/usr/lib/wsl/lib
 */
#include "wubu_kernel.h"
#include <stdio.h>

/* CUDA backend probe + register (compiled in wubu_kernel_cuda.cu).
 * We declare them as weak here so non-CUDA test builds link without
 * the CUDA object file. When wubu_kernel_cuda.o is linked, its strong
 * symbols override these. */
__attribute__((weak)) int wubu_cuda_backend_probe(void) { return 0; }
__attribute__((weak)) int wubu_cuda_backend_register(void) { return -1; }

/* g_use_gpu_backend: flag for proj_matmul in wubu_ssm.c.
 * Set to 1 when CUDA is detected, enabling GPU-quantized proj_matmul path. */
int g_use_gpu_backend = 0;

/* A:03 — CPU stub for proj_matmul_gpu. Returns 0 (CPU fallback).
 * In GPU builds, wubu_gpu_weight_cache.cu provides the real implementation
 * (strong symbol overrides this weak stub). */
__attribute__((weak))
int proj_matmul_gpu(const float *x, const uint8_t *W_q, int quant_type,
                    int n_rows, int n_cols, float *out) {
    (void)x; (void)W_q; (void)quant_type; (void)n_rows; (void)n_cols; (void)out;
    return 0; /* CPU fallback */
}

/* Device backend supports query: which kernel types does it handle? */
static int default_supports(wubu_kernel_type_t type) {
    (void)type;
    return 1;
}

/* Register device backends that are both compiled in AND available
 * at runtime. Called from wubu_kernel_init(). */
void wubu_kernel_register_backends(void) {
#if WUBU_HAS_CUDA
    int probed = wubu_cuda_backend_probe();
    fprintf(stderr, "[cuda] probe returned %d\n", probed);
    if (probed) {
        wubu_cuda_backend_register();
        g_use_gpu_backend = 1;
        fprintf(stderr, "[cuda] backend registered, g_use_gpu_backend=%d\n", g_use_gpu_backend);
    }
#endif

#if WUBU_HAS_VULKAN
    /* Vulkan compute backend — registration stub */
#endif

#if WUBU_HAS_METAL
    /* Metal backend — registration stub */
#endif

#if WUBU_HAS_ROCM
    /* ROCm backend — registration stub */
#endif

#if WUBU_HAS_BLAS
    /* BLAS backend — registration stub */
#endif
}
