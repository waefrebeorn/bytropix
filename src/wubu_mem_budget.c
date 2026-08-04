/*
 * wubu_mem_budget.c -- Memory budget calculator for OOM-proof inference.
 *
 * Self-contained C11. Reads /proc/meminfo for available RAM, computes
 * the safe KV cache size and forward buffer budget, never OOMs.
 */
#include "wubu_mem_budget.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- RAM detection (cross-platform) ---- */

#if defined(_WIN32)
#include <windows.h>
size_t wubu_mem_detect_available_ram(void) {
    /* Windows: use GlobalMemoryStatusEx — /proc/meminfo does not exist here. */
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) return 0;
    /* ullAvailPhys = physically installed memory available to processes. */
    return (size_t)ms.ullAvailPhys;
}
#else
size_t wubu_mem_detect_available_ram(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    size_t mem_available = 0;
    size_t mem_total = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "MemAvailable:", 13) == 0) {
            sscanf(line + 13, "%zu", &mem_available);
            mem_available *= 1024; /* KB to bytes */
        }
        if (strncmp(line, "MemTotal:", 9) == 0) {
            sscanf(line + 9, "%zu", &mem_total);
            mem_total *= 1024;
        }
    }
    fclose(f);
    /* If MemAvailable not found (old kernel), use 80% of MemTotal */
    if (mem_available == 0 && mem_total > 0)
        mem_available = (size_t)(mem_total * 0.8);
    return mem_available;
}
#endif /* !_WIN32 */

/* ---- Budget calculator ---- */

struct wubu_mem_budget {
    size_t available_ram;       /* bytes */
    size_t model_weight_sz;     /* bytes (mmap'd, counts as virtual not RSS) */
    int    n_gqa_layers;
    int    n_ssm_layers;
    int   *kv_dims;             /* array of n_gqa_layers kv_dims, or NULL */
    int    kv_dim_uniform;      /* used if kv_dims == NULL */
    int    bytes_per_kv_elem;   /* 2=F16, 4=F32, 1=Q8, 0=4KV(0.5) */
};

wubu_mem_budget_t *wubu_mem_budget_create(
    size_t available_ram,
    size_t model_weight_sz,
    int n_gqa_layers, int n_ssm_layers,
    const int *kv_dim_per_layer, int kv_dim_uniform,
    int bytes_per_kv_elem)
{
    wubu_mem_budget_t *b = (wubu_mem_budget_t *)calloc(1, sizeof(*b));
    if (!b) return NULL;

    if (available_ram == 0)
        available_ram = wubu_mem_detect_available_ram();
    b->available_ram = available_ram;
    b->model_weight_sz = model_weight_sz;
    b->n_gqa_layers = n_gqa_layers;
    b->n_ssm_layers = n_ssm_layers;
    b->bytes_per_kv_elem = bytes_per_kv_elem > 0 ? bytes_per_kv_elem : 2;

    if (kv_dim_per_layer && n_gqa_layers > 0) {
        b->kv_dims = (int *)malloc((size_t)n_gqa_layers * sizeof(int));
        if (b->kv_dims) memcpy(b->kv_dims, kv_dim_per_layer,
                               (size_t)n_gqa_layers * sizeof(int));
    } else {
        b->kv_dims = NULL;
        b->kv_dim_uniform = kv_dim_uniform;
    }
    return b;
}

/* Compute total KV elements across all GQA layers for ctx positions. */
static int64_t total_kv_elems(const wubu_mem_budget_t *b, int ctx) {
    int64_t total = 0;
    if (b->kv_dims) {
        for (int l = 0; l < b->n_gqa_layers; l++)
            total += (int64_t)ctx * b->kv_dims[l];
    } else {
        total = (int64_t)ctx * b->kv_dim_uniform * b->n_gqa_layers;
    }
    return total;
}

wubu_mem_budget_info_t wubu_mem_budget_compute(
    wubu_mem_budget_t *b,
    int requested_ctx,
    size_t ssm_state_sz,
    size_t forward_buf_sz,
    size_t moe_slot_sz,
    size_t gpu_weight_sz)
{
    wubu_mem_budget_info_t info;
    memset(&info, 0, sizeof(info));

    if (!b) return info;

    info.available_bytes = b->available_ram;
    info.model_weight_bytes = b->model_weight_sz;
    info.ssm_state_bytes = ssm_state_sz;
    info.forward_buf_bytes = forward_buf_sz;
    info.moe_slot_bytes = moe_slot_sz;
    info.gpu_weight_bytes = gpu_weight_sz;

    /* Model weights are mmap'd — they count as virtual memory, not RSS.
     * The OS pages in only what's accessed. So we don't subtract the full
     * model size from RAM; we subtract the *resident* portion, which is
     * roughly the layers we actively use + the embedding/output weights.
     * Conservative: subtract 40% of model size (active working set). */
    size_t model_rss_estimate = (size_t)(b->model_weight_sz * 0.40);

    /* Headroom: 10% of available RAM (safety margin for OS + malloc overhead). */
    info.headroom_bytes = b->available_ram / 10;

    /* Fixed costs: SSM states + MoE slots + GPU cache + headroom + model RSS */
    size_t fixed = ssm_state_sz + moe_slot_sz + gpu_weight_sz
                 + info.headroom_bytes + model_rss_estimate;

    /* Remaining for KV cache + forward buffers */
    size_t remaining = 0;
    if (b->available_ram > fixed)
        remaining = b->available_ram - fixed;

    /* Forward buffers are arena-allocated, reused per forward. They're
     * persistent (allocated once, not freed per token). Budget ~512MB. */
    size_t forward_budget = forward_buf_sz;
    if (forward_budget > remaining / 4)  /* cap at 25% of remaining */
        forward_budget = remaining / 4;
    info.forward_buf_bytes = forward_budget;
    if (remaining > forward_budget)
        remaining -= forward_budget;
    else
        remaining = 0;

    /* KV cache: remaining bytes / (2 * bytes_per_kv_elem) = total elements.
     * Factor 2 = K + V identical size. */
    int bpe = b->bytes_per_kv_elem;
    if (bpe < 1) bpe = 1;  /* 4KV uses 0.5 → treat as 1 byte (conservative) */

    /* total_kv_elems(ctx) * bpe * 2 (K+V) <= remaining
     * → ctx * total_kv_dim * bpe * 2 <= remaining
     * → ctx <= remaining / (total_kv_dim * bpe * 2) */
    int64_t total_kv_dim = 0;
    if (b->kv_dims) {
        for (int l = 0; l < b->n_gqa_layers; l++)
            total_kv_dim += b->kv_dims[l];
    } else {
        total_kv_dim = (int64_t)b->kv_dim_uniform * b->n_gqa_layers;
    }

    int max_ctx = 0;
    if (total_kv_dim > 0 && bpe > 0) {
        int64_t ctx_bytes_per_pos = total_kv_dim * (int64_t)bpe * 2;
        if (ctx_bytes_per_pos > 0)
            max_ctx = (int)(remaining / (size_t)ctx_bytes_per_pos);
    }

    /* Clamp to requested context */
    if (max_ctx > requested_ctx) max_ctx = requested_ctx;
    if (max_ctx < 64) max_ctx = 64;  /* minimum viable context */

    /* Actual KV cache bytes at max_ctx */
    info.kv_cache_bytes = total_kv_elems(b, max_ctx) * (int64_t)bpe * 2;
    info.max_kv_ctx = max_ctx;
    info.swa_window = (max_ctx < requested_ctx) ? max_ctx : 0;
    info.use_ssd_moe = (moe_slot_sz > 0) ? 1 : 0;

    /* AirLLM layer streaming: stream when the *requested* context's full KV
     * footprint would exceed available RAM (we had to shrink max_ctx below the
     * requested context because KV cache is RAM-bound). At 512K that means the
     * unconstrained KV (requested_ctx positions) is larger than RAM can hold,
     * so we must stream layers instead of materializing the whole model. */
    int64_t req_kv_elems = total_kv_elems(b, requested_ctx);
    int64_t req_kv_bytes  = req_kv_elems * (int64_t)bpe * 2;
    if (requested_ctx > max_ctx && req_kv_bytes > (int64_t)b->available_ram)
        info.use_layer_stream = 1;
    else
        info.use_layer_stream = 0;

    fprintf(stderr, "[membudget] RAM=%zuMB model=%zuMB ssm=%zuMB "
            "moe=%zuMB gpu=%zuMB fwd=%zuMB headroom=%zuMB → "
            "KV=%zuMB max_ctx=%d swa=%d\n",
            b->available_ram / (1024*1024),
            info.model_weight_bytes / (1024*1024),
            info.ssm_state_bytes / (1024*1024),
            info.moe_slot_bytes / (1024*1024),
            info.gpu_weight_bytes / (1024*1024),
            info.forward_buf_bytes / (1024*1024),
            info.headroom_bytes / (1024*1024),
            info.kv_cache_bytes / (1024*1024),
            info.max_kv_ctx, info.swa_window);

    return info;
}

void wubu_mem_budget_destroy(wubu_mem_budget_t *b) {
    if (!b) return;
    free(b->kv_dims);
    free(b);
}
