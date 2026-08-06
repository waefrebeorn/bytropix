/* wubu_backend.h — Backend dispatch interface for wubu_model_t.
 * Replaces #ifdef GPU_SUPPORT hardcoded paths in wubu_model.c,
 * wubu_moe.c, and wubu_ssm.c with a unified vtable-based dispatch.
 *
 * The backend is selected at model load time via the GGUF metadata
 * or runtime probe.  CPU-only builds link without any CUDA object;
 * GPU builds link wubu_model_gpu.o and set the backend pointer.
 *
 * ADR-002 opaque-struct seam: wubu_backend_t is an opaque pointer
 * stored in wubu_model_t.  Consumers include wubu_backend.h and
 * call wubu_backend_*() — never touch the struct fields directly.
 */

#ifndef WUBU_BACKEND_H
#define WUBU_BACKEND_H

#include "wubu_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Backend capability flags — OR'd into wubu_backend_t.capabilities. */
#define WUBU_BACKEND_CPU      0x01
#define WUBU_BACKEND_CUDA_GEMV 0x02
#define WUBU_BACKEND_CUDA_SSM  0x04
#define WUBU_BACKEND_CUDA_MOE  0x08
#define WUBU_BACKEND_CUDA_ATTN 0x10

/* Per-backend function pointers.  All take wubu_model_t* as the
 * first argument (the model owns the GPU context, KV cache, etc.). */
typedef struct wubu_backend_t {
    int capabilities;

    /* Initialise GPU resources for this model.
     * Returns 0 on success, -1 on failure (model->gpu_ctx stays NULL). */
    int (*init)(wubu_model_t *model, int max_ctx, int chunk_sz);

    /* Free GPU resources. */
    void (*free)(wubu_model_t *model);

    /* GQA attention forward for one layer.
     * GPU manages KV cache, projections, and attention internally.
     * Returns 0 on success, -1 on GPU failure (caller falls back to CPU). */
    int (*gqa_forward)(wubu_model_t *model, int layer_idx,
                       const float *h_norm, int C, float *h_attn);

    /* SSM forward for prefill (N>1 tokens, full GPU SSM).
     * Returns 0 on success, -1 on GPU failure. */
    int (*ssm_forward_prefill)(wubu_model_t *model, int layer_idx,
                               const float *h_norm, int C,
                               float *h_attn_out);

    /* SSM projection for hybrid decode (GPU quant matmuls + CPU recurrence).
     * Projects input to QKV and Z on GPU; CPU handles SSM recurrence.
     * Returns 0 on success, -1 on GPU failure. */
    int (*ssm_project)(wubu_model_t *model, int layer_idx,
                       const float *h_norm, int C,
                       float *qkv_out, float *z_out,
                       float *ssm_out_out);

    /* MoE expert dispatch.  Called from wubu_moe_forward() when
     * model->gpu_ctx is set and FORCE_CPU_MOE is not set. */
    void (*moe_experts)(const moe_weights_t *w,
                        const float *x_s,
                        const int *indices_s, const float *weights_s,
                        float *expert_contribs,
                        void *model_ptr);

    /* Quantised matmul (Q4_K / Q5_K / Q6_K).
     * Returns 0 on success, -1 on GPU failure.
     * stream is an opaque backend stream handle (CUDA stream for the
     * CUDA backend, NULL for CPU) — kept as void* so this agnostic
     * header never leaks CUDA types into CPU builds. */
    int (*quant_matmul)(const float *x, int n_rows, int n_cols,
                        const uint8_t *d_W_q, int quant_type,
                        float *y, void *stream);

    /* ---- SSM hybrid decode helpers (replaces direct GPU calls) ----
     * set_ssm_hybrid: point the layer's SSM at backend device buffers
     *   (d_ssm_state, d_q/k/v, stream) so the CPU recurrence kernel can
     *   read/write device memory directly. CPU backend: no-op.
     * sync_ssm_state_to_gpu: after CPU prefill, push CPU SSM+conv state
     *   to the device. CPU backend: no-op. */
    void (*set_ssm_hybrid)(wubu_model_t *model, int layer_idx,
                           struct ssm_layer_weights *ssm);
    void (*sync_ssm_state_to_gpu)(wubu_model_t *model, int layer_idx,
                                  const float *cpu_ssm_state,
                                  const float *cpu_conv_state);

    /* Max GPU chunk size for prefill (wubu_model_gpu_chunk_sz). */
    int (*chunk_size)(wubu_model_t *model);

    /* ---- ADR-003: KV cache is a file system ----
     * Namespace I/O routed through the backend so device-resident KV
     * (e.g. CUDA-managed KV pages) can be read/written by path without
     * a host round-trip. path is a /kv/... namespace path; n_floats is
     * the element count. Returns 0 on success, -1 on failure — the
     * caller falls back to the flat host tensor path. NULL entries
     * mean "no backend acceleration" (CPU fallback only). */
    int (*kvfs_read)(wubu_model_t *model, const char *path,
                     float *dst, size_t n_floats);
    int (*kvfs_write)(wubu_model_t *model, const char *path,
                      const float *src, size_t n_floats);
    /* Snapshot device-resident KV for a path (CUDA graph capture, etc).
     * NULL = not supported (caller uses the flat-tensor snapshot_json). */
    int (*kvfs_snapshot)(wubu_model_t *model, const char *path,
                         float *dst, size_t n_floats);
} wubu_backend_t;

/* Return the active backend for a model (NULL if CPU-only). */
static inline wubu_backend_t *wubu_backend_get(const wubu_model_t *model) {
    return model ? model->backend : NULL;
}

/* Check whether a backend supports a given capability. */
static inline bool wubu_backend_has(wubu_backend_t *backend, int cap) {
    return backend && (backend->capabilities & cap) != 0;
}

/* Dispatch helpers — these replace the #ifdef GPU_SUPPORT blocks
 * in wubu_model.c, wubu_moe.c, and wubu_ssm.c. */

/* GQA forward: try GPU, fall back to CPU on failure. */
static inline int wubu_backend_gqa_forward(wubu_model_t *model, int layer_idx,
                                           const float *h_norm, int C,
                                           float *h_attn)
{
    wubu_backend_t *backend = wubu_backend_get(model);
    if (backend && wubu_backend_has(backend, WUBU_BACKEND_CUDA_ATTN) && backend->gqa_forward) {
        int ok = backend->gqa_forward(model, layer_idx, h_norm, C, h_attn);
        if (ok == 0) return 0;
        /* GPU path failed — fall through to CPU */
    }
    /* CPU fallback is handled by the caller (wubu_gqa_forward). */
    return -1;  /* signal: caller should use CPU path */
}

/* SSM forward prefill: try GPU, fall back to CPU. */
static inline int wubu_backend_ssm_forward_prefill(wubu_model_t *model, int layer_idx,
                                                    const float *h_norm, int C,
                                                    float *h_attn_out)
{
    wubu_backend_t *backend = wubu_backend_get(model);
    if (backend && wubu_backend_has(backend, WUBU_BACKEND_CUDA_SSM) && backend->ssm_forward_prefill) {
        int ok = backend->ssm_forward_prefill(model, layer_idx, h_norm, C, h_attn_out);
        if (ok == 0) return 0;
    }
    return -1;  /* signal: caller should use CPU path */
}

/* SSM projection for hybrid decode: try GPU, fall back to CPU. */
static inline int wubu_backend_ssm_project(wubu_model_t *model, int layer_idx,
                                            const float *h_norm, int C,
                                            float *qkv_out, float *z_out,
                                            float *ssm_out_out)
{
    wubu_backend_t *backend = wubu_backend_get(model);
    if (backend && wubu_backend_has(backend, WUBU_BACKEND_CUDA_SSM) && backend->ssm_project) {
        int ok = backend->ssm_project(model, layer_idx, h_norm, C,
                                        qkv_out, z_out, ssm_out_out);
        if (ok == 0) return 0;
    }
    return -1;  /* signal: caller should use CPU path */
}

/* MoE expert dispatch: try GPU, fall back to CPU. */
static inline void wubu_backend_moe_experts(const moe_weights_t *w,
                                            const float *x_s,
                                            const int *indices_s, const float *weights_s,
                                            float *expert_contribs,
                                            void *model_ptr)
{
    wubu_model_t *model = (wubu_model_t *)model_ptr;
    wubu_backend_t *backend = wubu_backend_get(model);
    if (backend && wubu_backend_has(backend, WUBU_BACKEND_CUDA_MOE) && backend->moe_experts) {
        backend->moe_experts(w, x_s, indices_s, weights_s, expert_contribs, model_ptr);
        return;
    }
    /* CPU fallback is handled by the caller (wubu_moe_forward). */
}

/* ---- ADR-003: KV cache is a file system ----
 * Backend-routed namespace I/O. Returns 0 on success; -1 means "no
 * backend acceleration" — the caller (wubu_model_kvfs_read/write)
 * falls back to the flat host tensor through wubu_kvfs directly. */
static inline int wubu_backend_kvfs_read(wubu_model_t *model, const char *path,
                                         float *dst, size_t n_floats)
{
    wubu_backend_t *backend = wubu_backend_get(model);
    if (backend && backend->kvfs_read)
        return backend->kvfs_read(model, path, dst, n_floats);
    return -1;  /* signal: caller uses flat-tensor fallback */
}

static inline int wubu_backend_kvfs_write(wubu_model_t *model, const char *path,
                                          const float *src, size_t n_floats)
{
    wubu_backend_t *backend = wubu_backend_get(model);
    if (backend && backend->kvfs_write)
        return backend->kvfs_write(model, path, src, n_floats);
    return -1;  /* signal: caller uses flat-tensor fallback */
}

/* Handle-based backend routing: the backend may keep its own mapping
 * from handle to device-resident KV. Default: no acceleration (the
 * caller falls back to the flat host tensor through the handle).
 * h is a wubu_kvfs_handle_t* (opaque here — no header leak). */
static inline int wubu_backend_kvfs_handle_read(wubu_model_t *model,
                                                const void *h,
                                                float *dst, size_t n_floats)
{
    (void)model; (void)h; (void)dst; (void)n_floats;
    return -1;  /* CPU: caller uses the flat-tensor handle path */
}

static inline int wubu_backend_kvfs_handle_write(wubu_model_t *model,
                                                 const void *h,
                                                 const float *src,
                                                 size_t n_floats)
{
    (void)model; (void)h; (void)src; (void)n_floats;
    return -1;  /* CPU: caller uses the flat-tensor handle path */
}

/* SSM hybrid decode helpers — CPU no-ops, GPU routes to device buffers. */
static inline void wubu_backend_set_ssm_hybrid(wubu_model_t *model,
                                               int layer_idx,
                                               struct ssm_layer_weights *ssm)
{
    wubu_backend_t *backend = wubu_backend_get(model);
    if (backend && backend->set_ssm_hybrid)
        backend->set_ssm_hybrid(model, layer_idx, ssm);
}

static inline void wubu_backend_sync_ssm_state_to_gpu(wubu_model_t *model,
                                                      int layer_idx,
                                                      const float *cpu_ssm_state,
                                                      const float *cpu_conv_state)
{
    wubu_backend_t *backend = wubu_backend_get(model);
    if (backend && backend->sync_ssm_state_to_gpu)
        backend->sync_ssm_state_to_gpu(model, layer_idx,
                                       cpu_ssm_state, cpu_conv_state);
}

/* Max GPU chunk size for prefill; 0 means no GPU chunking. */
static inline int wubu_backend_chunk_size(wubu_model_t *model)
{
    wubu_backend_t *backend = wubu_backend_get(model);
    if (backend && backend->chunk_size)
        return backend->chunk_size(model);
    return 0;
}

/* Backend vtable constructors. The CPU backend is always available
 * (wubu_backend.c); the CUDA backend is linked only in GPU builds
 * (wubu_backend_cuda.c). Model init installs the CPU backend by
 * default; successful GPU init swaps in the CUDA backend. */
wubu_backend_t *wubu_backend_cpu_get(void);
wubu_backend_t *wubu_backend_cuda_get(void);

#ifdef __cplusplus
}
#endif

#endif // WUBU_BACKEND_H