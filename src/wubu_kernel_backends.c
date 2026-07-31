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

/* CUDA backend probe + register (compiled only when WUBU_HAS_CUDA=1) */
#if WUBU_HAS_CUDA
extern int wubu_cuda_backend_probe(void);
extern int wubu_cuda_backend_register(void);
#endif

/* Device backend supports query: which kernel types does it handle? */
static int default_supports(wubu_kernel_type_t type) {
    (void)type;
    return 1;
}

/* Register device backends that are both compiled in AND available
 * at runtime. Called from wubu_kernel_init(). */
void wubu_kernel_register_backends(void) {
#if WUBU_HAS_CUDA
    if (wubu_cuda_backend_probe()) {
        wubu_cuda_backend_register();
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
