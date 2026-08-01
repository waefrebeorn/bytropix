/*
 * wubu_mem_budget.h -- Memory budget calculator for OOM-proof inference.
 *
 * Principle (ds4-ssd / airllm): never pre-allocate more than available RAM.
 * Compute the memory budget from available system RAM, subtract fixed costs
 * (model weights, SSM states), then cap KV cache + forward buffers to fit.
 *
 * Kevin-Bacon convergence: decode is memory-bandwidth-bound; the arena
 * allocator (C01/I01) bounds peak temp; the budget system bounds peak
 * persistent. Together: no OOM, ever.
 *
 * Self-contained C11. No god headers. Opaque struct.
 */
#ifndef WUBU_MEM_BUDGET_H
#define WUBU_MEM_BUDGET_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque budget handle. */
typedef struct wubu_mem_budget wubu_mem_budget_t;

/* Budget breakdown — returned by wubu_mem_budget_compute. */
typedef struct {
    size_t available_bytes;     /* system RAM available for inference */
    size_t model_weight_bytes;   /* GGUF/safetensors blob size (mmap'd) */
    size_t ssm_state_bytes;      /* SSM + conv states (persistent) */
    size_t kv_cache_bytes;       /* K+V cache allocation (persistent) */
    size_t forward_buf_bytes;    /* per-forward temp buffers (arena) */
    size_t moe_slot_bytes;       /* MoE expert slot-bank (persistent) */
    size_t gpu_weight_bytes;     /* GPU weight cache (device memory) */
    size_t headroom_bytes;       /* safety margin (10% of available) */

    int    max_kv_ctx;           /* computed max KV cache positions */
    int    swa_window;           /* SWA window if max_kv_ctx < requested */
    int    use_ssd_moe;           /* 1 if SSD MoE paging needed */
    int    use_layer_stream;     /* 1 if airllm layer streaming needed */
} wubu_mem_budget_info_t;

/* Create a budget calculator.
 *   available_ram   = total system RAM available (bytes); 0 = auto-detect
 *   model_weight_sz  = size of the model file on disk (bytes)
 *   n_gqa_layers     = number of GQA (attention) layers
 *   n_ssm_layers     = number of SSM layers
 *   kv_dim_per_layer = array of kv_dim for each GQA layer (or NULL if uniform)
 *   kv_dim_uniform   = kv_dim if kv_dim_per_layer is NULL
 *   bytes_per_kv_elem = 2 (F16) or 4 (F32) or 1 (Q8) or 0.5 (4KV)
 */
wubu_mem_budget_t *wubu_mem_budget_create(
    size_t available_ram,
    size_t model_weight_sz,
    int n_gqa_layers, int n_ssm_layers,
    const int *kv_dim_per_layer, int kv_dim_uniform,
    int bytes_per_kv_elem);

/* Compute the budget for a requested max context.
 *   requested_ctx   = desired max context (e.g. 524288 for 512K)
 *   ssm_state_sz    = SSM + conv state total bytes
 *   forward_buf_sz  = per-forward temp buffer bytes (5 * N * d_model * 4)
 *   moe_slot_sz     = MoE slot-bank bytes (0 if no MoE)
 *   gpu_weight_sz   = GPU weight cache bytes (0 if no GPU)
 * Returns filled info struct. max_kv_ctx may be < requested_ctx. */
wubu_mem_budget_info_t wubu_mem_budget_compute(
    wubu_mem_budget_t *b,
    int requested_ctx,
    size_t ssm_state_sz,
    size_t forward_buf_sz,
    size_t moe_slot_sz,
    size_t gpu_weight_sz);

/* Destroy the budget calculator. */
void wubu_mem_budget_destroy(wubu_mem_budget_t *b);

/* Utility: detect available system RAM (reads /proc/meminfo on Linux).
 * Returns bytes available for user processes (MemAvailable). */
size_t wubu_mem_detect_available_ram(void);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MEM_BUDGET_H */
