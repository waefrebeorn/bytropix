#ifndef WUBU_MODEL_GPU_H
#define WUBU_MODEL_GPU_H

/*
 * wubu_model_gpu.h — the GPU-accelerated forward path API.
 *
 * Extracted from wubu_model.h (Strangler Fig, ADR-002): consumers that
 * only need GPU offload functions can include this header instead of the
 * full model header. The GPU functions take wubu_model_t* / ssm / moe
 * handles and never need the model struct layout, so we forward-declare
 * the types instead of including the full definitions.
 *
 * Implemented in wubu_model_gpu.cu. Falls back to CPU when no GPU.
 */

#include "wubu_model_fwd.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize GPU context: upload GQA weights, allocate KV cache + scratch.
 * max_ctx: maximum KV cache positions (e.g. 262144).
 * chunk_sz: max tokens per GPU batch (e.g. 512).
 * Returns 1 on success, 0 on failure.
 * When GPU context is active, wubu_model_forward() automatically uses
 * GPU for GQA attention layers. */
int wubu_model_gpu_init(wubu_model_t *model, int max_ctx, int chunk_sz);

/* Run one GQA layer on GPU. Internal: called by wubu_model_forward
 * when gpu_ctx != NULL. */
int wubu_model_gpu_gqa_forward(wubu_model_t *model, int layer_idx,
                                const float *h_norm, int C, float *h_attn);

/* Get GPU chunk size (max tokens per batched GPU call). 0 if GPU off. */
int wubu_model_gpu_chunk_sz(wubu_model_t *model);

/* Run SSM projections (qkv, gate) on GPU via quantized matmul kernels.
 * h_norm: [C, D_MODEL] input; C: number of tokens (1 for decode);
 * qkv_out: [C, CONV_DIM] output (host); z_out: [C, VALUE_DIM] (host). */
int wubu_model_gpu_ssm_project(wubu_model_t *model, int layer_idx,
                                const float *h_norm, int C,
                                float *qkv_out, float *z_out,
                                float *ssm_out_out);

/* Run GPU SSM completely on GPU: quantized matmuls → conv1d → SiLU →
 * split → L2 norm → recurrence → gated norm → ssm_out projection.
 * Returns 1 on success, 0 on fallback to CPU. */
int wubu_model_gpu_ssm_forward_full(wubu_model_t *model, int layer_idx,
                                     const float *h_norm, int C,
                                     float *h_attn_out);

/* Set SSM layer GPU pointers from gpu_ctx for hybrid (CPU SSM + GPU
 * recurrence). Called by wubu_model_forward fallback paths when gpu_ctx
 * exists. gpu_ctx is model->gpu_ctx (void*), ssm is layer->ssm to fill. */
void wubu_gpu_set_ssm_hybrid(void *gpu_ctx, int layer_idx, ssm_layer_weights *ssm);

/* Sync CPU SSM state + conv state to GPU before forward_full decode. */
void wubu_gpu_sync_ssm_state_to_gpu(void *gpu_ctx, int layer_idx,
                                     const float *cpu_ssm_state,
                                     const float *cpu_conv_state);

/* Sync GPU SSM state + conv state back to CPU after forward_full decode. */
void wubu_gpu_sync_ssm_state_to_cpu(void *gpu_ctx, int layer_idx,
                                     float *cpu_ssm_state,
                                     float *cpu_conv_state);

/* Run MoE experts via GPU kernel, replacing CPU quantized matmul loop.
 * Shared expert and router remain on CPU. Called per-token from
 * wubu_moe_forward's expert loop. */
void wubu_model_gpu_moe_experts(const moe_weights_t *w,
    const float *x_s,
    const int *indices_s, const float *weights_s,
    float *expert_contribs,
    void *model_ptr);

/* Free all GPU resources and reset gpu_ctx to NULL. */
void wubu_model_gpu_free(wubu_model_t *model);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MODEL_GPU_H */
