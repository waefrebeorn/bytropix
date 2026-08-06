/* wubu_backend.c — CPU-only backend stub.
 * Provides the wubu_backend_t vtable for CPU-only builds.
 * GPU builds override this with wubu_backend_cuda.c which
 * populates the vtable with CUDA function pointers.
 *
 * ADR-003: the CPU backend exposes kvfs_read/kvfs_write as NULL
 * (no device-resident KV), so namespace I/O falls back to the flat
 * host tensor through wubu_kvfs — the speed-kernel path.
 */

#include "wubu_backend.h"

static wubu_backend_t wubu_backend_cpu = {
    .capabilities = WUBU_BACKEND_CPU,
    .init = NULL,
    .free = NULL,
    .gqa_forward = NULL,
    .ssm_forward_prefill = NULL,
    .ssm_project = NULL,
    .moe_experts = NULL,
    .quant_matmul = NULL,
    .set_ssm_hybrid = NULL,
    .sync_ssm_state_to_gpu = NULL,
    .chunk_size = NULL,
    .kvfs_read = NULL,
    .kvfs_write = NULL,
    .kvfs_snapshot = NULL,
};

/* Return the CPU-only backend.  Called by wubu_model_init
 * when no GPU is available or GPU_SUPPORT is not defined. */
wubu_backend_t *wubu_backend_cpu_get(void) {
    return &wubu_backend_cpu;
}
