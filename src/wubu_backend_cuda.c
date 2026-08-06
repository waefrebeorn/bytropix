/* wubu_backend_cuda.c — CUDA backend for wubu_backend_t vtable.
 * Populates the vtable with CUDA function pointers when GPU_SUPPORT
 * is defined.  This replaces the #ifdef GPU_SUPPORT blocks in
 * wubu_model.c, wubu_moe.c, and wubu_ssm.c with a unified dispatch.
 */

#include "wubu_backend.h"
#include "wubu_model_gpu.h"

static int cuda_backend_init(wubu_model_t *model, int max_ctx, int chunk_sz) {
    return wubu_model_gpu_init(model, max_ctx, chunk_sz);
}

static void cuda_backend_free(wubu_model_t *model) {
    wubu_model_gpu_free(model);
}

static int cuda_backend_gqa_forward(wubu_model_t *model, int layer_idx,
                                     const float *normed, int B, int T,
                                     float *attn_out,
                                     void *k_cache, void *v_cache, int cache_len,
                                     float *k_out, float *v_out,
                                     int head_dim, int q_heads, int kv_heads) {
    return wubu_model_gpu_gqa_forward(model, layer_idx, normed, B, T, attn_out,
                                       k_cache, v_cache, cache_len,
                                       k_out, v_out, head_dim, q_heads, kv_heads);
}

static int cuda_backend_ssm_forward(wubu_model_t *model, int layer_idx,
                                     const float *normed, int B, int T,
                                     float *out, void *ssm_state, void *conv_state) {
    return wubu_model_gpu_ssm_forward(model, layer_idx, normed, B, T, out,
                                       ssm_state, conv_state);
}

static int cuda_backend_ssm_forward_prefill(wubu_model_t *model, int layer_idx,
                                             const float *normed, int B, int T,
                                             float *attn_out) {
    return wubu_model_gpu_ssm_forward_full(model, layer_idx, normed, B, T, attn_out);
}

static void cuda_backend_moe_experts(const moe_weights_t *w,
                                     const float *x_s,
                                     const int *indices_s, const float *weights_s,
                                     float *expert_contribs,
                                     void *model_ptr) {
    wubu_model_gpu_moe_experts(w, x_s, indices_s, weights_s, expert_contribs, model_ptr);
}

static int cuda_backend_quant_matmul(const float *x, int n_rows, int n_cols,
                                      const uint8_t *q_weight, int q_type,
                                      int q_group_size, float *out) {
    (void)q_group_size;
    return wubu_model_gpu_quant_matmul(x, n_rows, n_cols, q_weight, q_type, out);
}

static void cuda_backend_ssm_sync_to_gpu(void *gpu_ctx, int layer_idx, void *ssm) {
    wubu_gpu_set_ssm_hybrid(gpu_ctx, layer_idx, (ssm_layer_weights *)ssm);
}

static void cuda_backend_ssm_sync_to_cpu(void *gpu_ctx, int layer_idx, void *ssm) {
    (void)gpu_ctx; (void)layer_idx; (void)ssm;
    /* GPU→CPU sync is handled inside wubu_model_gpu.cu via the
     * wubu_gpu_sync_ssm_state_to_cpu() function.  The backend
     * pointer is stored in the model's gpu_ctx; callers use
     * wubu_gpu_sync_ssm_state_to_cpu() directly when needed. */
}

static void cuda_backend_set_ssm_hybrid(void *gpu_ctx, int layer_idx, void *ssm) {
    wubu_gpu_set_ssm_hybrid(gpu_ctx, layer_idx, (ssm_layer_weights *)ssm);
}

static wubu_backend_t wubu_backend_cuda = {
    .capabilities = WUBU_BACKEND_CPU | WUBU_BACKEND_CUDA_GEMV | WUBU_BACKEND_CUDA_SSM |
                    WUBU_BACKEND_CUDA_MOE | WUBU_BACKEND_CUDA_ATTN,
    .init = cuda_backend_init,
    .free = cuda_backend_free,
    .gqa_forward = cuda_backend_gqa_forward,
    .ssm_forward = cuda_backend_ssm_forward,
    .ssm_forward_prefill = cuda_backend_ssm_forward_prefill,
    .moe_experts = cuda_backend_moe_experts,
    .quant_matmul = cuda_backend_quant_matmul,
    .ssm_sync_to_gpu = cuda_backend_ssm_sync_to_gpu,
    .ssm_sync_to_cpu = cuda_backend_ssm_sync_to_cpu,
    .set_ssm_hybrid = cuda_backend_set_ssm_hybrid,
};

wubu_backend_t *wubu_backend_cuda_get(void) {
    return &wubu_backend_cuda;
}