/**
 * wubu-cuda.cu — WuBu hyperbolic operations as CUDA kernels
 * following llama.cpp's ggml-cuda pattern.
 *
 * Each op follows the pattern:
 *   .cu = kernel definition + host dispatch function
 *   .cuh = declarations + launch config
 *   common.cuh = shared helpers
 *
 * Ops:
 *   GGML_OP_WUBU_POINCARE_EXP  — exponential map
 *   GGML_OP_WUBU_POINCARE_LOG  — logarithmic map 
 *   GGML_OP_WUBU_MOE_TOP2      — top-2 MoE routing
 *   GGML_OP_WUBU_ROLLING_HASH  — rolling hash (SimpleHash)
 */

#include "ggml-cuda/common.cuh"
#include "ggml.h"

/* ─── Kernel: Poincaré Exponential Map ─── */
/* Each thread handles one element of one vector.
 * v: [B, N] input tangent vectors
 * y: [B, N] output Poincaré ball points
 * c: curvature, s: scale
 */
__global__ void wubu_exp_map_kernel(
    const float* __restrict__ v,
    float* __restrict__ y,
    const int B, const int N,
    const float c, const float s)
{
    const int bid = blockIdx.x;    /* batch index */
    const int tid = threadIdx.x;   /* element within vector */
    
    if (bid >= B || tid >= N) return;
    
    const float* v_batch = v + bid * N;
    float* y_batch = y + bid * N;
    
    /* Compute norm across the vector (shared memory reduction) */
    __shared__ float s_norm[256];
    float local_sum = v_batch[tid] * v_batch[tid];
    s_norm[tid] = local_sum;
    __syncthreads();
    
    /* Reduce in shared memory */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_norm[tid] += s_norm[tid + s];
        __syncthreads();
    }
    
    float norm = sqrtf(s_norm[0] + 1e-7f);
    
    if (norm < 1e-7f) {
        y_batch[tid] = 0.0f;
        return;
    }
    
    float sqrt_c = sqrtf(fmaxf(c, 1e-7f));
    float scaled_radius = s * sqrt_c * norm;
    float tanh_val = tanhf(scaled_radius);
    float lambda = tanh_val / (sqrt_c * norm + 1e-7f);
    
    float val = lambda * v_batch[tid];
    
    /* Project into ball */
    float max_norm = (1.0f / sqrt_c) * (1.0f - 1e-7f);
    if (fabsf(lambda * norm) > max_norm) {
        val *= max_norm / (fabsf(lambda * norm));
    }
    
    y_batch[tid] = val;
}

/* ─── Kernel: Top-2 MoE Routing ─── */
/* Each token selects its top-2 experts from N experts.
 * logits: [B, N] routing weights
 * expert_idx: [B, 2] output selected expert indices
 * expert_w: [B, 2] output routing weights (softmax over top-2)
 */
__global__ void wubu_moe_top2_kernel(
    const float* __restrict__ logits,
    int* __restrict__ expert_idx,
    float* __restrict__ expert_w,
    const int B, const int N)
{
    const int bid = blockIdx.x;
    if (bid >= B) return;
    
    const float* row = logits + bid * N;
    
    /* Linear scan for top-2 (N is small: 8-256 experts) */
    int best1 = 0, best2 = 1;
    float val1 = row[0], val2 = row[1];
    if (val2 > val1) {
        float tmp = val1; val1 = val2; val2 = tmp;
        int t = best1; best1 = best2; best2 = t;
    }
    
    for (int i = 2; i < N; i++) {
        float v = row[i];
        if (v > val1) {
            val2 = val1; best2 = best1;
            val1 = v;    best1 = i;
        } else if (v > val2) {
            val2 = v;    best2 = i;
        }
    }
    
    expert_idx[bid * 2 + 0] = best1;
    expert_idx[bid * 2 + 1] = best2;
    
    /* Softmax over top-2 */
    float maxv = fmaxf(val1, val2);
    float e1 = expf(val1 - maxv);
    float e2 = expf(val2 - maxv);
    float sum = e1 + e2 + 1e-10f;
    expert_w[bid * 2 + 0] = e1 / sum;
    expert_w[bid * 2 + 1] = e2 / sum;
}

/* ─── Host Dispatch Functions ─── */

void ggml_cuda_op_wubu_exp_map(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    const ggml_tensor* src = dst->src[0];
    const float c = ((float*)dst->op_params)[0];
    const float s = ((float*)dst->op_params)[1];
    
    const int B = (int)src->ne[1];  /* batch */
    const int N = (int)src->ne[0];  /* dim */
    
    float* d_src = (float*)src->data;
    float* d_dst = (float*)dst->data;
    
    dim3 block(min(N, 256));
    dim3 grid(B);
    
    wubu_exp_map_kernel<<<grid, block, 0, ctx.stream>>>(
        d_src, d_dst, B, N, c, s
    );
}

void ggml_cuda_op_wubu_moe_top2(ggml_backend_cuda_context& ctx, ggml_tensor* dst) {
    const ggml_tensor* src = dst->src[0];
    
    const int B = (int)src->ne[1];
    const int N = (int)src->ne[0];
    
    float* d_logits = (float*)src->data;
    int* d_idx = (int*)dst->data;
    float* d_w = (float*)((char*)dst->data + B * 2 * sizeof(int));
    
    wubu_moe_top2_kernel<<<B, 1, 0, ctx.stream>>>(
        d_logits, d_idx, d_w, B, N
    );
}

/* ─── Dispatch Table Entry ─── */
/* This would be registered in ggml_cuda_compute_forward switch:
 *
 * case GGML_OP_WUBU_POINCARE_EXP:
 *     ggml_cuda_op_wubu_exp_map(ctx, dst);
 *     break;
 * case GGML_OP_WUBU_MOE_TOP2:
 *     ggml_cuda_op_wubu_moe_top2(ctx, dst);
 *     break;
 */
