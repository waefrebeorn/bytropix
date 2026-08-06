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
     * Returns 0 on success, -1 on GPU failure (caller falls back to CPU). */
    int (*gqa_forward)(wubu_model_t *model, int layer_idx,
                       const float *normed, int B, int T,
                       float *attn_out,
                       void *k_cache, void *v_cache, int cache_len,
                       float *k_out, float *v_out,
                       int head_dim, int q_heads, int kv_heads);

    /* SSM forward for one layer (single-token decode path).
     * Returns 0 on success, -1 on GPU failure. */
    int (*ssm_forward)(wubu_model_t *model, int layer_idx,
                       const float *normed, int B, int T,
                       float *out, void *ssm_state, void *conv_state);

    /* SSM forward for prefill (N>1 tokens).
     * Returns 0 on success, -1 on GPU failure. */
    int (*ssm_forward_prefill)(wubu_model_t *model, int layer_idx,
                               const float *normed, int B, int T,
                               float *attn_out);

    /* MoE expert dispatch.  Called from wubu_moe_forward() when
     * model->gpu_ctx is set and FORCE_CPU_MOE is not set. */
    void (*moe_experts)(const moe_weights_t *w,
                        const float *x_s,
                        const int *indices_s, const float *weights_s,
                        float *expert_contribs,
                        void *model_ptr);

    /* Quantised matmul (Q4_K / Q5_K / Q6_K).  Used by the CPU
     * fallback path when GPU quant_matmul is unavailable. */
    int (*quant_matmul)(const float *x, int n_rows, int n_cols,
                        const uint8_t *q_weight, int q_type,
                        int q_group_size, float *out);

    /* Sync SSM state between CPU and GPU (for speculative decode
     * rollback / checkpoint-restore). */
    void (*ssm_sync_to_gpu)(void *gpu_ctx, int layer_idx, void *ssm);
    void (*ssm_sync_to_cpu)(void *gpu_ctx, int layer_idx, void *ssm);

    /* Set the SSM hybrid mode (GPU vs CPU recurrence) for a layer. */
    void (*set_ssm_hybrid)(void *gpu_ctx, int layer_idx, void *ssm);
} wubu_backend_t;

/* Return the active backend for a model (NULL if CPU-only). */
static inline wubu_backend_t *wubu_backend_get(const wubu_model_t *model) {
    return (wubu_backend_t *)model->gpu_ctx;
}

/* Check whether a backend supports a given capability. */
static inline bool wubu_backend_has(wubu_backend_t *backend, int cap) {
    return backend && (backend->capabilities & cap) != 0;
}

/* Dispatch helpers — these replace the #ifdef GPU_SUPPORT blocks
 * in wubu_model.c, wubu_moe.c, and wubu_ssm.c. */

/* GQA forward: try GPU, fall back to CPU on failure. */
static inline int wubu_backend_gqa_forward(wubu_model_t *model, int layer_idx,
                                           const float *normed, int B, int T,
                                           float *attn_out,
                                           void *k_cache, void *v_cache, int cache_len,
                                           float *k_out, float *v_out,
                                           int head_dim, int q_heads, int kv_heads)
{
    wubu_backend_t *backend = wubu_backend_get(model);
    if (backend && wubu_backend_has(backend, WUBU_BACKEND_CUDA_ATTN) && backend->gqa_forward) {
        int ok = backend->gqa_forward(model, layer_idx, normed, B, T,
                                       attn_out, k_cache, v_cache, cache_len,
                                       k_out, v_out, head_dim, q_heads, kv_heads);
        if (ok == 0) return 0;
        /* GPU path failed — fall through to CPU */
    }
    /* CPU fallback is handled by the caller (wubu_gqa_forward). */
    return -1;  /* signal: caller should use CPU path */
}

/* SSM forward (single-token decode): try GPU, fall back to CPU. */
static inline int wubu_backend_ssm_forward(wubu_model_t *model, int layer_idx,
                                           const float *normed, int B, int T,
                                           float *out, void *ssm_state, void *conv_state)
{
    wubu_backend_t *backend = wubu_backend_get(model);
    if (backend && wubu_backend_has(backend, WUBU_BACKEND_CUDA_SSM) && backend->ssm_forward) {
        int ok = backend->ssm_forward(model, layer_idx, normed, B, T,
                                       out, ssm_state, conv_state);
        if (ok == 0) return 0;
    }
    return -1;  /* signal: caller should use CPU path */
}

/* SSM forward prefill: try GPU, fall back to CPU. */
static inline int wubu_backend_ssm_forward_prefill(wubu_model_t *model, int layer_idx,
                                                    const float *normed, int B, int T,
                                                    float *attn_out)
{
    wubu_backend_t *backend = wubu_backend_get(model);
    if (backend && wubu_backend_has(backend, WUBU_BACKEND_CUDA_SSM) && backend->ssm_forward_prefill) {
        int ok = backend->ssm_forward_prefill(model, layer_idx, normed, B, T, attn_out);
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

#ifdef __cplusplus
}
#endif

#endif // WUBU_BACKEND_H