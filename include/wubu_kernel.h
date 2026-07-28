/*
 * wubu_kernel.h -- WuBuOS-agnostic compute kernel strategy.
 *
 * Mandate (from the Colonel project): kernels must be usable on ALL
 * accelerators ("all accelerators acceptable"). This rules out Triton, which
 * is CUDA-locked. Our design is ISA-portable C + intrinsics with a backend
 * registry:
 *
 *   - CPU backend  : OUR OWN tiled GEMM (wubu_gemm.c) — AVX2-FMA / AVX512-FMA,
 *                    cache-blocked, with a portable scalar fallback. No external
 *                    BLAS dependency.
 *   - Device backend: CUDA / Metal / Vulkan / ROCm register a function pointer
 *                    via wubu_gemm_register_device() at init. They win if present.
 *
 * The single entry point wubu_gemm_f32() dispatches at runtime. New accelerators
 * drop in a .c/.cu/.metal that calls wubu_gemm_register_device() — no changes
 * to caller code (quantized_matmul, moe_expert_forward_lib, etc.).
 *
 * Why not Triton: Triton compiles to CUDA PTX only; it cannot target Metal/
 * Vulkan/ROCm. That contradicts the WuBuOS all-accelerator requirement. Our
 * hand-written kernels + backend registry cover every target with one API.
 */
#ifndef WUBU_KERNEL_H
#define WUBU_KERNEL_H

#include "wubu_gemm.h"   /* the concrete implementation + registry */

#ifdef __cplusplus
extern "C" {
#endif

/* Accelerator families WuBuOS targets. A device backend advertises which it
 * serves; the CPU backend always covers "cpu". */
typedef enum {
    WUBU_DEV_CPU    = 0,
    WUBU_DEV_CUDA   = 1,
    WUBU_DEV_METAL  = 2,
    WUBU_DEV_VULKAN = 3,
    WUBU_DEV_ROCM   = 4
} wubu_device_kind_t;

/* Register a device backend. fn has the wubu_gemm_f32 signature. Returns 0
 * if accepted. A registered device takes precedence over the CPU kernel. */
static inline int wubu_kernel_register_device(wubu_gemm_fn fn, const char *name) {
    return wubu_gemm_register_device(fn, name);
}

/* Which kernel backend is active (for diagnostics). */
static inline const char *wubu_kernel_active(void) {
    return wubu_gemm_active_backend();
}

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KERNEL_H */
