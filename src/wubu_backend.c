/* wubu_backend.c — CPU-only backend stub.
 * Provides the wubu_backend_t vtable for CPU-only builds.
 * GPU builds override this with wubu_backend_cuda.c which
 * populates the vtable with CUDA function pointers.
 */

#include "wubu_backend.h"

static wubu_backend_t wubu_backend_cpu = {
    .capabilities = WUBU_BACKEND_CPU,
    .init = NULL,
    .free = NULL,
    .gqa_forward = NULL,
    .ssm_forward = NULL,
    .ssm_forward_prefill = NULL,
    .moe_experts = NULL,
    .quant_matmul = NULL,
    .ssm_sync_to_gpu = NULL,
    .ssm_sync_to_cpu = NULL,
    .set_ssm_hybrid = NULL,
};

/* Return the CPU-only backend.  Called by wubu_model_init
 * when no GPU is available or GPU_SUPPORT is not defined. */
wubu_backend_t *wubu_backend_cpu_get(void) {
    return &wubu_backend_cpu;
}