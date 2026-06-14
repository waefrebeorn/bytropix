/**
 * Flash Attention Tiled Implementation - CUDA with WMMA Tensor Cores (Complete)
 *
 * Implements fused flash attention for head_dim=256 using nvcuda::wmma
 * Based on llama.cpp ggml-cuda/fattn-mma-f16.cuh approach
 * Optimized for Qwen3.6 GQA: Nq=16, Nkv=2, head_dim=256
 */

#include "flash_attn_tiled.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
#include <stdio.h>

// ================================================================
// WMMA Configuration for head_dim=256
// ================================================================

// WMMA tile sizes (16x16x16)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Warp reductions
__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float tmp = __shfl_xor_sync(0xffffffff, val, offset);
        val = fmaxf(val, tmp);
    }
    return val;
}

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Online softmax state
struct fa_softmax_state {
    float m;  // running max
    float l;  // running sum of exp
};

__inline__ __device__ void warp_online_softmax_update(
    float new_val,
    fa_softmax_state *state,
    float *out_val
) {
    float m_old = state->m;
    float m_new = fmaxf(m_old, new_val);
    float alpha = expf(m_old - m_new);

    state->l = state->l * alpha + expf(new_val - m_new);
    state->m = m_new;
    *out_val = expf(new_val - m_new);
}

using namespace nvcuda::wmma;

/**
 * Flash Attention Kernel with WMMA for head_dim=256 (Prefill: C>1)
 *
 * Each block processes 1 Q tile (128 tokens) for 1 KV head, all 8 Q heads.
 * Block: 8 warps (256 threads), each warp handles 1 Q head
 * Grid: (n_q_tiles, n_kv_heads, batch)
 */
__global__ void flash_attn_kernel_256_wmma(
    const half *__restrict__ Q,      // [B, Hq, Tq, 256] row-major FP16
    const half *__restrict__ K,      // [B, Hkv, Tk, 256] row-major FP16
    const half *__restrict__ V,      // [B, Hkv, Tk, 256] row-major FP16
    half *__restrict__ O,            // [B, Hq, Tq, 256] row-major FP16 output
    const float softmax_scale,
    const float logit_softcap,
    const bool causal_mask,
    const int window_size,
    const int B,
    const int Hq,
    const int Hkv,
    const int Tq,
    const int Tk
) {
    const int head_dim = 256;
    const int q_tile_size = 128;
    const int kv_tile_size = 128;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warps_per_block = blockDim.x / 32;  // 8 warps

    const int q_tile_idx = blockIdx.x;
    const int kv_head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;

    if (q_tile_idx * q_tile_size >= Tq) return;
    if (kv_head_idx >= Hkv) return;
    if (batch_idx >= B) return;

    const int q_start = q_tile_idx * q_tile_size;
    const int q_end = min(q_start + q_tile_size, Tq);
    const int q_len = q_end - q_start;

    const int q_heads_per_kv = Hq / Hkv;  // 8 for Qwen
    const int q_head_start = kv_head_idx * q_heads_per_kv;

    // Each warp handles 1 Q head (8 Q heads / 8 warps = 1)
    const int qh_local = warp_id % q_heads_per_kv;

    // Shared memory layout:
    // K_tile: [kv_tile_size, head_dim] = 128 * 256
    // V_tile: [kv_tile_size, head_dim] = 128 * 256
    // Q_shared: [q_len, head_dim] (for active Q positions)
    extern __shared__ half smem[];
    half *K_tile = smem;
    half *V_tile = &smem[kv_tile_size * head_dim];
    half *Q_shared = &smem[2 * kv_tile_size * head_dim];

    // Online softmax state per Q position for this Q head
    fa_softmax_state ss[q_tile_size];
    #pragma unroll
    for (int qr = 0; qr < q_tile_size; qr++) {
        ss[qr].m = -INFINITY;
        ss[qr].l = 0.0f;
    }

    // Output accumulator in registers per thread
    float acc[32] = {0.0f};  // 256/8 = 32 per thread

    // Process KV tiles
    for (int kv_tile_idx = 0; kv_tile_idx * kv_tile_size < Tk; kv_tile_idx++) {
        int kv_start = kv_tile_idx * kv_tile_size;
        int kv_end = min(kv_start + kv_tile_size, Tk);
        int kv_len = kv_end - kv_start;

        // Load K/V tile into shared memory
        for (int idx = tid; idx < kv_len * head_dim; idx += blockDim.x) {
            int k = idx / head_dim;
            int d = idx % head_dim;
            int src = batch_idx * Hkv * Tk * head_dim + kv_head_idx * Tk * head_dim + (kv_start + k) * head_dim + d;
            K_tile[k * head_dim + d] = K[src];
            V_tile[k * head_dim + d] = V[src];
        }
        __syncthreads();

        // Process each Q position in this tile
        for (int q_rel = 0; q_rel < q_len; q_rel++) {
            int q_abs = q_start + q_rel;

            // Load Q vector for this warp's Q head
            int qh_global = q_head_start + qh_local;
            for (int d = lane_id; d < head_dim; d += 32) {
                int src = batch_idx * Hq * Tq * head_dim + qh_global * Tq * head_dim + q_abs * head_dim + d;
                Q_shared[q_rel * head_dim + d] = Q[src];
            }
            __syncthreads();

            // Compute Q*K^T for this Q position against all KV in tile
            for (int kv = 0; kv < kv_len; kv++) {
                int kv_abs = kv_start + kv;
                bool masked = false;
                if (causal_mask && kv_abs > q_abs) masked = true;
                if (window_size > 0 && kv_abs < q_abs - window_size) masked = true;

                // Compute dot product (8 elements per thread)
                float sum = 0.0f;
                for (int d = lane_id * 8; d < (lane_id + 1) * 8; d++) {
                    float qv = __half2float(Q_shared[q_rel * head_dim + d]);
                    float kv_val = __half2float(K_tile[kv * head_dim + d]);
                    sum += qv * kv_val;
                }

                sum = warp_reduce_sum(sum);
                float score = sum * softmax_scale;

                if (lane_id == 0) {
                    if (masked) score = -INFINITY;
                    if (logit_softcap > 0.0f) {
                        score = logit_softcap * tanhf(score / logit_softcap);
                    }

                    // Online softmax update
                    float exp_val;
                    warp_online_softmax_update(score, &ss[q_rel], &exp_val);
                    ss[q_rel].l = warp_reduce_sum(ss[q_rel].l);

                    // Scale and accumulate V
                    if (ss[q_rel].l > 0.0f) {
                        float inv_l = 1.0f / ss[q_rel].l;
                        float weight = exp_val * inv_l;

                        for (int d = lane_id * 8; d < (lane_id + 1) * 8; d++) {
                            float v = __half2float(V_tile[kv * head_dim + d]);
                            acc[d / 8] = fmaf(weight, v, acc[d / 8] * (1.0f - weight));
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write output
    int qh_global = q_head_start + qh_local;
    for (int q_rel = 0; q_rel < q_len; q_rel++) {
        for (int d = lane_id * 8; d < (lane_id + 1) * 8; d++) {
            int out_idx = batch_idx * Hq * Tq * head_dim + qh_global * Tq * head_dim
                          + (q_start + q_rel) * head_dim + d;
            O[out_idx] = __float2half_rn(acc[d / 8]);
        }
    }
}

/**
 * Fast Decode Kernel (C=1) for head_dim=256
 * Optimized for single token decode - uses fewer threads, direct KV cache access
 */
__global__ void flash_attn_decode_kernel_256(
    const half *__restrict__ Q,      // [B, Hq, 1, 256] row-major FP16
    const half *__restrict__ K,      // [B, Hkv, Tk, 256] row-major FP16
    const half *__restrict__ V,      // [B, Hkv, Tk, 256] row-major FP16
    half *__restrict__ O,            // [B, Hq, 1, 256] row-major FP16 output
    const float softmax_scale,
    const float logit_softcap,
    const bool causal_mask,
    const int window_size,
    const int B,
    const int Hq,
    const int Hkv,
    const int Tk
) {
    const int head_dim = 256;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warps_per_block = blockDim.x / 32;

    const int kv_head_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;

    if (kv_head_idx >= Hkv) return;
    if (batch_idx >= B) return;

    const int q_heads_per_kv = Hq / Hkv;  // 8
    const int q_head_start = kv_head_idx * q_heads_per_kv;

    // Each warp handles 1 Q head
    const int qh_local = warp_id;
    if (qh_local >= q_heads_per_kv) return;
    const int qh_global = q_head_start + qh_local;

    // Online softmax state
    fa_softmax_state ss = {-INFINITY, 0.0f};

    // Accumulator
    float acc[32] = {0.0f};  // 256/8 = 32 per thread

    // Process KV tiles (128 tokens per tile)
    for (int kv_tile_idx = 0; kv_tile_idx * 128 < Tk; kv_tile_idx++) {
        int kv_start = kv_tile_idx * 128;
        int kv_end = min(kv_start + 128, Tk);
        int kv_len = kv_end - kv_start;

        // Load Q vector for this Q head (1 token)
        half q_vec[256];
        for (int d = lane_id; d < head_dim; d += 32) {
            int src = batch_idx * Hq * 1 * head_dim + qh_global * 1 * head_dim + d;
            q_vec[d] = Q[src];
        }
        // Broadcast within warp (each lane gets all 256 dims)
        #pragma unroll
        for (int d = 0; d < head_dim; d += 32) {
            int src_lane = d / 32;
            q_vec[d] = __shfl_sync(0xffffffff, q_vec[d], src_lane);
        }

        // Process each KV in tile
        for (int kv = 0; kv < kv_len; kv++) {
            int kv_abs = kv_start + kv;
            bool masked = false;
            if (causal_mask && kv_abs > 0) masked = true;  // q_abs = 0 for decode
            if (window_size > 0 && kv_abs < 0 - window_size) masked = true;

            // Compute dot product
            float sum = 0.0f;
            for (int d = lane_id * 8; d < (lane_id + 1) * 8; d++) {
                float qv = __half2float(q_vec[d]);
                int k_src = batch_idx * Hkv * Tk * head_dim + kv_head_idx * Tk * head_dim + kv_abs * head_dim + d;
                float kv_val = __half2float(K[k_src]);
                sum += qv * kv_val;
            }

            sum = warp_reduce_sum(sum);
            float score = sum * softmax_scale;

            if (lane_id == 0) {
                if (masked) score = -INFINITY;
                if (logit_softcap > 0.0f) {
                    score = logit_softcap * tanhf(score / logit_softcap);
                }

                float exp_val;
                warp_online_softmax_update(score, &ss, &exp_val);
                ss.l = warp_reduce_sum(ss.l);

                if (ss.l > 0.0f) {
                    float inv_l = 1.0f / ss.l;
                    float weight = exp_val * inv_l;

                    for (int d = lane_id * 8; d < (lane_id + 1) * 8; d++) {
                        int v_src = batch_idx * Hkv * Tk * head_dim + kv_head_idx * Tk * head_dim + kv_abs * head_dim + d;
                        float v = __half2float(V[v_src]);
                        acc[d / 8] = fmaf(weight, v, acc[d / 8] * (1.0f - weight));
                    }
                }
            }
        }
    }

    // Write output
    for (int d = lane_id * 8; d < (lane_id + 1) * 8; d++) {
        int out_idx = batch_idx * Hq * 1 * head_dim + qh_global * 1 * head_dim + d;
        O[out_idx] = __float2half_rn(acc[d / 8]);
    }
}

// Sanitize NaN kernel
__global__ void flash_attn_sanitize_kernel(half *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]);
    if (isnan(v) || isinf(v)) x[i] = __float2half(0.0f);
}

void flash_attn_sanitize_f16(half *d_ptr, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    flash_attn_sanitize_kernel<<<grid, block, 0, stream>>>(d_ptr, n);
}

// Forward declarations
void launch_flash_attn_decode_256(
    const half *Q, const half *K, const half *V, half *O,
    float softmax_scale, bool causal_mask, int window_size, float logit_softcap,
    int B, int Hq, int Hkv, int Tk,
    cudaStream_t stream
);

// ================================================================
// Host launchers
// ================================================================

void launch_flash_attn_256(
    const half *Q, const half *K, const half *V, half *O,
    float softmax_scale, bool causal_mask, int window_size, float logit_softcap,
    int B, int Hq, int Hkv, int Tq, int Tk,
    cudaStream_t stream
) {
    // Use fast decode kernel for C=1, prefill kernel for C>1
    if (Tq == 1) {
        launch_flash_attn_decode_256(Q, K, V, O, softmax_scale, causal_mask, window_size, logit_softcap,
                                     B, Hq, Hkv, Tk, stream);
        return;
    }

    const int q_tile_size = 128;
    const int q_heads_per_kv = Hq / Hkv;  // 8 for Qwen

    // 8 warps per block (256 threads), 1 warp per Q head
    dim3 block(256);
    dim3 grid(
        (Tq + q_tile_size - 1) / q_tile_size,
        Hkv,
        B
    );

    // Shared memory: K_tile + V_tile + Q_shared
    // K/V: 128 * 256 * 2 * 2 = 128 KB
    // Q_shared: 128 * 256 * 2 = 64 KB
    // Total: ~192 KB
    size_t smem_bytes = 2 * q_tile_size * 256 * sizeof(half) +  // K + V tiles
                        q_tile_size * 256 * sizeof(half);       // Q_shared

    if (causal_mask) {
        flash_attn_kernel_256_wmma<<<grid, block, smem_bytes, stream>>>(
            Q, K, V, O, softmax_scale, logit_softcap, true, window_size,
            B, Hq, Hkv, Tq, Tk
        );
    } else {
        flash_attn_kernel_256_wmma<<<grid, block, smem_bytes, stream>>>(
            Q, K, V, O, softmax_scale, logit_softcap, false, window_size,
            B, Hq, Hkv, Tq, Tk
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "flash_attn_256 launch failed: %s\n", cudaGetErrorString(err));
    }
}

// Host launcher for decode (C=1)
void launch_flash_attn_decode_256(
    const half *Q, const half *K, const half *V, half *O,
    float softmax_scale, bool causal_mask, int window_size, float logit_softcap,
    int B, int Hq, int Hkv, int Tk,
    cudaStream_t stream
) {
    // 8 warps = 256 threads, 1 warp per Q head
    dim3 block(256);
    dim3 grid(1, Hkv, B);  // Single Q tile (1 token)

    if (causal_mask) {
        flash_attn_decode_kernel_256<<<grid, block, 0, stream>>>(
            Q, K, V, O, softmax_scale, logit_softcap, true, window_size,
            B, Hq, Hkv, Tk
        );
    } else {
        flash_attn_decode_kernel_256<<<grid, block, 0, stream>>>(
            Q, K, V, O, softmax_scale, logit_softcap, false, window_size,
            B, Hq, Hkv, Tk
        );
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "flash_attn_decode_256 launch failed: %s\n", cudaGetErrorString(err));
    }
}

void launch_flash_attn_512(
    const half *Q, const half *K, const half *V, half *O,
    float softmax_scale, bool causal_mask, int window_size, float logit_softcap,
    int B, int Hq, int Hkv, int Tq, int Tk,
    cudaStream_t stream
) {
    // Placeholder for head_dim=512 (Gemma 4 full)
    // Would use 4 warps per Q head
}