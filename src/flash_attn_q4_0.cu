/**
 * Q4_0 Unified Flash Attention Kernel - Prefill + Decode
 *
 * Based on llama.cpp ggml-cuda/fattn-vec.cuh pattern:
 * - Single kernel for both prefill (C>1) and decode (C=1)
 * - Template parameter `cols_per_block` = 1 for decode, = q_tile_size/warps for prefill
 * - Grid-stride over KV tiles
 * - Warp-level online softmax with VKQ register accumulation
 * - Cross-KV-tile reduction via shared memory
 * - Uses dp4a instruction for Q4_0 K dot product (like llama.cpp)
 */

#include "flash_attn_q4_0.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <float.h>

// KV_BLOCK_SIZE for PagedAttention (matching wubu_model.h)
#define KV_BLOCK_SIZE 16

// Q4_0 constants matching llama.cpp GGML
#define QK4_0 32
#define QI4_0 8
#define QI8_1 2

// Half max for softmax initialization (avoid NaN on subtraction)
#define HALF_MAX_HALF __float2half(65504.0f/2)
#define SOFTMAX_FTZ_THRESHOLD -20.0f
#define FATTN_KQ_MAX_OFFSET (3.0f * 0.6931f)

// Q4_0 block structure matching llama.cpp
typedef struct { uint16_t d; uint8_t qs[QK4_0/2]; } block_q4_0;

// Q2_0 block structure (for V cache - TurboQuant+ asymmetric)
// 32 elements, fp16 scale, 2-bit signed [-2, 1]
// 10 bytes per 32 elements
typedef struct {
    uint16_t d;      // fp16 scale
    uint8_t qs[8];   // 32 × 2-bit values (4 per byte)
} block_q2_0;

// Warp reduction helpers
__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float tmp = __shfl_xor_sync(0xffffffff, val, offset);
        val = fmaxf(val, tmp);
    }
    return val;
}

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Quantize fp16 Q to q8_1 (int8 with per-8-element scale) - like llama.cpp
// Each warp quantizes its Q vectors for its heads
// Q_reg: [head_dim] fp16 per Q head (from shared memory)
// Output: Q_q8 [head_dim/4] int32 (4 int8 packed), Q_ds [head_dim/8] float2 (scale, sum) per 8 elements
__inline__ __device__ void quantize_Q_to_q8_1(const half *Q_reg, int *Q_q8, float2 *Q_ds, int head_dim, int lane_id) {
    for (int k_KQ_0 = lane_id; k_KQ_0 < head_dim / 4; k_KQ_0 += QI8_1) {
        int idx = k_KQ_0 * 4;
        half2 h2_0 = *((const half2*)(Q_reg + idx));
        half2 h2_1 = *((const half2*)(Q_reg + idx + 2));
        float2 f2_0 = __half22float2(h2_0);
        float2 f2_1 = __half22float2(h2_1);
        float vals[4] = {f2_0.x, f2_0.y, f2_1.x, f2_1.y};

        // Share values with partner thread (lane_id ^ 1) for 8-element reduction
        float vals_shared[8];
        vals_shared[0] = vals[0]; vals_shared[1] = vals[1]; vals_shared[2] = vals[2]; vals_shared[3] = vals[3];
        vals_shared[4] = __shfl_xor_sync(0xffffffff, vals[0], 1);
        vals_shared[5] = __shfl_xor_sync(0xffffffff, vals[1], 1);
        vals_shared[6] = __shfl_xor_sync(0xffffffff, vals[2], 1);
        vals_shared[7] = __shfl_xor_sync(0xffffffff, vals[3], 1);

        // Find max abs value for quantization across 8 elements
        float amax = fabsf(vals_shared[0]);
        float sum = vals_shared[0];
        for (int j = 1; j < 8; j++) {
            amax = fmaxf(amax, fabsf(vals_shared[j]));
            sum += vals_shared[j];
        }

        // Quantize 4 elements (this thread's portion)
        float scale = amax / 127.0f;
        if (scale == 0.0f) scale = 1.0f;

        int8_t q8[4];
        for (int j = 0; j < 4; j++) {
            q8[j] = (int8_t)roundf(vals[j] / scale);
        }

        // Pack into int32
        int q32 = *(int*)q8;
        Q_q8[k_KQ_0] = q32;

        // Only one thread per pair writes the scale/sum
        if ((lane_id & 1) == 0) {
            Q_ds[k_KQ_0 / QI8_1] = make_float2(scale, sum);
        }
    }
}

// dp4a-based Q4_0 K dot product (matching llama.cpp vec_dot_fattn_vec_KQ_q4_0 exactly)
// Q_q8: [head_dim/4] int32 (packed int8), Q_ds: [head_dim/8] float2 (scale, sum)
// K_block: Q4_0 blocks
__inline__ __device__ float vec_dot_q4_0_dp4a(const uint8_t *K_block, const int *Q_q8, const float2 *Q_ds, int head_dim, int lane_id) {
    const block_q4_0 *K_q4 = (const block_q4_0 *)K_block;
    float sum = 0.0f;

    for (int k_KQ_0 = lane_id; k_KQ_0 < head_dim / 4; k_KQ_0 += 32) {
        const int ib    = (k_KQ_0 * 4) / QK4_0;
        const int iqs4  = (k_KQ_0 * 4) % (QK4_0/2);
        const int shift = ((k_KQ_0 * 4) % QK4_0) / (QK4_0/2);

        int v;
        const uint8_t *src = K_q4[ib].qs + sizeof(int)*iqs4;
        v = *(const int*)src;
        v = (v >> (4*shift)) & 0x0F0F0F0F;

        // Convert K nibbles (0-15) to int8 (-8..7)
        int k_int8 = v - 0x08080808;

        // DP4A: sum of 4 products of int8
        int u = Q_q8[k_KQ_0];
        int sumi = __dp4a(u, k_int8, 0);

        const float2 Q_ds_val = Q_ds[k_KQ_0 / QI8_1];
        float k_scale = __half2float(K_q4[ib].d);

        // llama.cpp formula: sum += k_scale * (sumi * Q_ds.x - (8/QI8_1) * Q_ds.y)
        sum += k_scale * (sumi * Q_ds_val.x - 4.0f * Q_ds_val.y);
    }
    return warp_reduce_sum(sum);
}

// Q2_0 V dequantization - returns float2 for vectorized access (TurboQuant+ asymmetric)
// V cache uses Q2_0 (2-bit signed [-2, 1])
__inline__ __device__ float2 dequant_q2_0_V_float2(const uint8_t *V_block, int d) {
    const block_q2_0 *V_q2 = (const block_q2_0 *)V_block;
    int bk = d / 32;
    int el = d % 32;
    const uint8_t *vp = V_q2[bk].qs;
    float scale = __half2float(*(const __half*)V_q2[bk].d);
    int q0 = (vp[el / 4] >> (2 * (el % 4))) & 0x3;
    int q1 = (vp[(el + 1) / 4] >> (2 * ((el + 1) % 4))) & 0x3;
    q0 -= 2; q1 -= 2;  // [-2, 1]
    return make_float2(scale * (float)q0, scale * (float)q1);
}

// Q4_0 V dequantization (legacy - for FP16 path)
__inline__ __device__ float2 dequant_q4_0_V_float2(const uint8_t *V_block, int d) {
    int bk = d / 32;
    int el = d % 32;
    const uint8_t *vp = V_block + (size_t)bk * 18;
    float scale = __half2float(*(const __half*)vp);
    int q0 = (vp[2 + el/2] >> (4 * (el % 2))) & 0xF;
    int q1 = (vp[2 + (el+1)/2] >> (4 * ((el+1) % 2))) & 0xF;
    q0 -= 8; q1 -= 8;
    return make_float2(scale * (float)q0, scale * (float)q1);
}

/**
 * Unified Q4_0 Flash Attention Kernel - PREFILL
 *
 * Template parameters:
 *   COLS_PER_BLOCK = q_tile_size/4 for prefill (128/4=32) to match 128 threads
 *   HEAD_DIM = 256 (Qwen)
 *   N_Q_HEADS = 16
 *   N_KV_HEADS = 2
 *   Q_HEADS_PER_KV = 8
 *   BLOCK_THREADS = 128 (4 warps)
 *   KV_TILE_SIZE = 128
 */
template<
    int COLS_PER_BLOCK,
    int HEAD_DIM,
    int N_Q_HEADS,
    int N_KV_HEADS,
    int Q_HEADS_PER_KV,
    int BLOCK_THREADS,
    int KV_TILE_SIZE
>
__global__ void flash_attn_q4_0_prefill_kernel(
    const half *__restrict__ Q,           // [B, Hq, Tq, HEAD_DIM] FP16
    const uint8_t *__restrict__ K_blocks, // [B, Hkv, Tk, HEAD_DIM] Q4_0
    const uint8_t *__restrict__ V_blocks, // [B, Hkv, Tk, HEAD_DIM] Q4_0
    half *__restrict__ O,                 // [B, Hq, Tq, HEAD_DIM] FP16
    const float softmax_scale,
    const bool causal_mask,
    const int window_size,
    const int B,
    const int Tq,
    const int Tk,
    const int q_tile_size
) {
    const int head_dim = HEAD_DIM;
    const int n_q = N_Q_HEADS;
    const int n_kv = N_KV_HEADS;
    const int ncols = Q_HEADS_PER_KV;  // 8
    const int warp_size = 32;
    const int n_threads = BLOCK_THREADS;  // 128
    const int n_warps = n_threads / warp_size;  // 4

    const int tid = threadIdx.x;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;

    // Grid mapping:
    // blockIdx.x = q_tile_idx
    // blockIdx.y = kv_tile_idx (parallel KV chunks)
    // blockIdx.z = batch_idx * n_kv * (ncols/n_warps) + kv_head_idx * (ncols/n_warps) + qh_group
    const int q_tile_idx = blockIdx.x;
    const int kv_tile_idx = blockIdx.y;
    const int batch_kv_qh = blockIdx.z;

    const int qh_groups = ncols / n_warps;  // 8/4 = 2
    const int batch_idx = batch_kv_qh / (n_kv * qh_groups);
    const int residual = batch_kv_qh % (n_kv * qh_groups);
    const int kv_head_idx = residual / qh_groups;
    const int qh_group = residual % qh_groups;  // 0..1

    if (q_tile_idx * q_tile_size >= Tq) return;
    if (kv_head_idx >= n_kv) return;
    if (batch_idx >= B) return;

    const int q_start = q_tile_idx * q_tile_size;
    const int q_end = min(q_start + q_tile_size, Tq);
    const int q_len = q_end - q_start;

    const int kv_start = kv_tile_idx * KV_TILE_SIZE;
    const int kv_end = min(kv_start + KV_TILE_SIZE, Tk);
    const int kv_len = kv_end - kv_start;

    if (kv_len <= 0) return;

    // Each warp handles qh_per_warp = ncols/n_warps = 2 Q heads
    const int qh_per_warp = ncols / n_warps;
    const int qh_base = qh_group * n_warps + warp_id * qh_per_warp;
    if (qh_base >= ncols) return;

    // Shared memory allocation:
    extern __shared__ char smem[];
    size_t offset = 0;
    half *Q_smem = (half*)(smem + offset);
    offset += n_warps * qh_per_warp * HEAD_DIM * sizeof(half);
    int *Q_q8 = (int*)(smem + offset);
    offset += n_warps * qh_per_warp * (HEAD_DIM / 4) * sizeof(int);
    float2 *Q_ds = (float2*)(smem + offset);
    offset += n_warps * qh_per_warp * (HEAD_DIM / 8) * sizeof(float2);
    // KQ_max/sum: q_len * n_warps * qh_per_warp
    float *KQ_max_shared = (float*)(smem + offset);
    offset += q_len * n_warps * qh_per_warp * sizeof(float);
    float *KQ_sum_shared = (float*)(smem + offset);

    // Load Q into shared memory for our Q heads
    for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
        const int qh_global = kv_head_idx * ncols + qh_base + qh_idx;
        for (int d = tid; d < head_dim; d += n_threads) {
            int src = batch_idx * n_q * Tq * head_dim + qh_global * Tq * head_dim + q_start * head_dim + d;
            Q_smem[warp_id * qh_per_warp * HEAD_DIM + qh_idx * HEAD_DIM + d] = Q[src];
        }
    }
    __syncthreads();

    // Quantize Q from shared memory to q8_1 for our Q heads
    for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
        half *Q_reg = Q_smem + warp_id * qh_per_warp * HEAD_DIM + qh_idx * HEAD_DIM;
        int *q_q8 = Q_q8 + warp_id * qh_per_warp * (HEAD_DIM / 4) + qh_idx * (HEAD_DIM / 4);
        float2 *q_ds = Q_ds + warp_id * qh_per_warp * (HEAD_DIM / 8) + qh_idx * (HEAD_DIM / 8);
        quantize_Q_to_q8_1(Q_reg, q_q8, q_ds, head_dim, lane_id);
    }
    __syncthreads();

    // Process each Q position in this tile
    for (int q_rel = 0; q_rel < q_len; q_rel++) {
        int q_abs = q_start + q_rel;

        // Per-Q-position softmax state (matching llama.cpp online softmax with V accumulation)
        float KQ_max[2] = {-FLT_MAX/2.0f, -FLT_MAX/2.0f};
        float KQ_sum[2] = {0.0f, 0.0f};
        // VKQ accumulation: each thread handles 2 elements per head
        float VKQ[2][2] = {{0.0f, 0.0f}};

        // First pass: Process KV tiles, compute KQ, online softmax, accumulate VKQ
        for (int kv = 0; kv < kv_len; kv++) {
            int kv_abs = kv_start + kv;

            bool masked = false;
            if (causal_mask && kv_abs > q_abs) masked = true;
            if (window_size > 0 && kv_abs < q_abs - window_size) masked = true;

            if (!masked) {
                const uint8_t *K_block_base = K_blocks + (size_t)batch_idx * n_kv * Tk * head_dim / 32 * 18;
                K_block_base += kv_head_idx * Tk * head_dim / 32 * 18;
                K_block_base += kv_abs * head_dim / 32 * 18;

                const uint8_t *V_block_base = V_blocks + (size_t)batch_idx * n_kv * Tk * head_dim / 32 * 18;
                V_block_base += kv_head_idx * Tk * head_dim / 32 * 18;
                V_block_base += kv_abs * head_dim / 32 * 18;

                // Compute KQ for our Q heads using dp4a
                float KQ_reg[2] = {0.0f, 0.0f};
                for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
                    int *q_q8 = Q_q8 + warp_id * qh_per_warp * (HEAD_DIM / 4) + qh_idx * (HEAD_DIM / 4);
                    float2 *q_ds = Q_ds + warp_id * qh_per_warp * (HEAD_DIM / 8) + qh_idx * (HEAD_DIM / 8);
                    KQ_reg[qh_idx] = vec_dot_q4_0_dp4a(K_block_base, q_q8, q_ds, head_dim, lane_id);
                    KQ_reg[qh_idx] *= softmax_scale;
                }

                // Online softmax + V accumulation (matching llama.cpp exactly)
                for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
                    float score = KQ_reg[qh_idx];
                    if (isnan(score)) score = -INFINITY;

                    float m_new = fmaxf(KQ_max[qh_idx] + FATTN_KQ_MAX_OFFSET, score + FATTN_KQ_MAX_OFFSET) - FATTN_KQ_MAX_OFFSET;
                    float alpha = expf(KQ_max[qh_idx] - m_new);
                    float exp_score = expf(score - m_new);

                    // Update online softmax state
                    KQ_sum[qh_idx] = KQ_sum[qh_idx] * alpha + exp_score;
                    KQ_max[qh_idx] = m_new;

                    // Accumulate V (each thread handles 2 elements per head)
                    for (int e = 0; e < 2; e++) {
                        int d = tid + e * (n_threads / 2);
                        if (d < head_dim) {
                            // Use Q2_0 V dequantization for TurboQuant+ asymmetric path
                            float2 v_val = dequant_q2_0_V_float2(V_block_base, d);
                            VKQ[qh_idx][e] = VKQ[qh_idx][e] * alpha + exp_score * v_val.x;
                            if (d + 1 < head_dim) {
                                VKQ[qh_idx][e] = VKQ[qh_idx][e] * alpha + exp_score * v_val.y;
                            }
                        }
                    }
                }
            }
        }

        // Cross-KV-tile reduction: store per-tile KQ_max and KQ_sum
        int local_idx = q_rel * n_warps * qh_per_warp + warp_id * qh_per_warp;
        for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
            KQ_max_shared[local_idx + qh_idx] = KQ_max[qh_idx];
            KQ_sum_shared[local_idx + qh_idx] = KQ_sum[qh_idx];
        }
        __syncthreads();

        // Reduce across KV tiles (only warp 0 lane 0 does this)
        if (warp_id == 0 && lane_id == 0) {
            for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
                float m = -FLT_MAX/2.0f;
                float l = 0.0f;
                for (int t = 0; t < gridDim.y; t++) {
                    int idx = t * q_len * n_warps * qh_per_warp + q_rel * n_warps * qh_per_warp + qh_idx;
                    float m_t = KQ_max_shared[idx] + FATTN_KQ_MAX_OFFSET;
                    float l_t = KQ_sum_shared[idx];
                    float m_new = fmaxf(m, m_t);
                    float alpha = expf(m - m_new);
                    l = l * alpha + l_t * expf(m_t - m_new);
                    m = m_new;
                }
                KQ_max_shared[local_idx + qh_idx] = m - FATTN_KQ_MAX_OFFSET;
                KQ_sum_shared[local_idx + qh_idx] = l;
            }
        }
        __syncthreads();

        // Rescale VKQ with final l = KQ_sum_final
        for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
            float l_final = KQ_sum_shared[local_idx + qh_idx];
            if (l_final > 0.0f) {
                float scale = KQ_sum[qh_idx] / l_final;
                for (int e = 0; e < 2; e++) {
                    VKQ[qh_idx][e] *= scale;
                }
            }
        }
        __syncthreads();

        // Write output: each thread writes 2 elements per head
        for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
            const int qh_global = kv_head_idx * ncols + qh_base + qh_idx;
            for (int e = 0; e < 2; e++) {
                int d = tid + e * (n_threads / 2);
                if (d < head_dim) {
                    int out_idx = batch_idx * n_q * Tq * head_dim + qh_global * Tq * head_dim + q_abs * head_dim + d;
                    O[out_idx] = __float2half_rn(VKQ[qh_idx][e]);
                }
            }
        }
    }
}

/**
 * Unified Q4_0 Flash Attention Kernel - DECODE ONLY (COLS_PER_BLOCK=1)
 *
 * For decode: C=1, single Q position, grid-stride over KV tiles
 * Each warp handles 2 Q heads (since 4 warps * 2 = 8 Q heads per KV head)
 */
template<
    int HEAD_DIM,
    int N_Q_HEADS,
    int N_KV_HEADS,
    int Q_HEADS_PER_KV,
    int BLOCK_THREADS,
    int KV_TILE_SIZE
>
__global__ void flash_attn_q4_0_decode_kernel(
    const half *__restrict__ Q,           // [B, Hq, Tq, HEAD_DIM] FP16, Tq=1
    const uint8_t *__restrict__ K_blocks, // [B, Hkv, Tk, HEAD_DIM] Q4_0
    const uint8_t *__restrict__ V_blocks, // [B, Hkv, Tk, HEAD_DIM] Q4_0
    half *__restrict__ O,                 // [B, Hq, Tq, HEAD_DIM] FP16
    const float softmax_scale,
    const bool causal_mask,
    const int window_size,
    const int B,
    const int Tq,
    const int Tk,
    const int q_tile_size
) {
    const int head_dim = HEAD_DIM;
    const int n_q = N_Q_HEADS;
    const int n_kv = N_KV_HEADS;
    const int ncols = Q_HEADS_PER_KV;  // 8
    const int warp_size = 32;
    const int n_threads = BLOCK_THREADS;  // 128
    const int n_warps = n_threads / warp_size;  // 4

    const int tid = threadIdx.x;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;

    // Grid mapping:
    // blockIdx.x = 0 (only 1 Q tile for decode)
    // blockIdx.y = kv_tile_idx (parallel KV chunks)
    // blockIdx.z = batch_idx * n_kv * (ncols/n_warps) + kv_head_idx * (ncols/n_warps) + qh_group
    const int kv_tile_idx = blockIdx.y;
    const int batch_kv_qh = blockIdx.z;

    const int qh_groups = ncols / n_warps;  // 8/4 = 2
    const int batch_idx = batch_kv_qh / (n_kv * qh_groups);
    const int residual = batch_kv_qh % (n_kv * qh_groups);
    const int kv_head_idx = residual / qh_groups;
    const int qh_group = residual % qh_groups;  // 0..1

    if (kv_head_idx >= n_kv) return;
    if (batch_idx >= B) return;

    const int kv_start = kv_tile_idx * KV_TILE_SIZE;
    const int kv_end = min(kv_start + KV_TILE_SIZE, Tk);
    const int kv_len = kv_end - kv_start;

    if (kv_len <= 0) return;

    // Each warp handles qh_per_warp = ncols/n_warps = 2 Q heads
    const int qh_per_warp = ncols / n_warps;
    const int qh_base = qh_group * n_warps + warp_id * qh_per_warp;
    if (qh_base >= ncols) return;

    // Shared memory allocation:
    extern __shared__ char smem[];
    size_t offset = 0;
    half *Q_smem = (half*)(smem + offset);
    offset += n_warps * qh_per_warp * HEAD_DIM * sizeof(half);
    int *Q_q8 = (int*)(smem + offset);
    offset += n_warps * qh_per_warp * (HEAD_DIM / 4) * sizeof(int);
    float2 *Q_ds = (float2*)(smem + offset);
    offset += n_warps * qh_per_warp * (HEAD_DIM / 8) * sizeof(float2);
    float *KQ_max_shared = (float*)(smem + offset);
    offset += n_warps * qh_per_warp * sizeof(float);
    float *KQ_sum_shared = (float*)(smem + offset);

    // Load Q into shared memory for our Q heads (C=1, so q_start=0)
    for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
        const int qh_global = kv_head_idx * ncols + qh_base + qh_idx;
        for (int d = tid; d < head_dim; d += n_threads) {
            int src = batch_idx * n_q * Tq * head_dim + qh_global * Tq * head_dim + 0 * head_dim + d;
            Q_smem[warp_id * qh_per_warp * HEAD_DIM + qh_idx * HEAD_DIM + d] = Q[src];
        }
    }
    __syncthreads();

    // Quantize Q from shared memory to q8_1 for our Q heads
    for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
        half *Q_reg = Q_smem + warp_id * qh_per_warp * HEAD_DIM + qh_idx * HEAD_DIM;
        int *q_q8 = Q_q8 + warp_id * qh_per_warp * (HEAD_DIM / 4) + qh_idx * (HEAD_DIM / 4);
        float2 *q_ds = Q_ds + warp_id * qh_per_warp * (HEAD_DIM / 8) + qh_idx * (HEAD_DIM / 8);
        quantize_Q_to_q8_1(Q_reg, q_q8, q_ds, head_dim, lane_id);
    }
    __syncthreads();

    // Single Q position (C=1, q_rel=0)
    int q_abs = 0;  // Only 1 token in decode

    // Per-Q-position softmax state (matching llama.cpp online softmax with V accumulation)
    float KQ_max[2] = {-FLT_MAX/2.0f, -FLT_MAX/2.0f};
    float KQ_sum[2] = {0.0f, 0.0f};
    // VKQ accumulation: each thread handles 2 elements per head
    float VKQ[2][2] = {{0.0f, 0.0f}};

    // First pass: Process KV tiles, compute KQ, online softmax, accumulate VKQ
    for (int kv = 0; kv < kv_len; kv++) {
        int kv_abs = kv_start + kv;

        bool masked = false;
        if (causal_mask && kv_abs > q_abs) masked = true;
        if (window_size > 0 && kv_abs < q_abs - window_size) masked = true;

        if (!masked) {
            const uint8_t *K_block_base = K_blocks + (size_t)batch_idx * n_kv * Tk * head_dim / 32 * 18;
            K_block_base += kv_head_idx * Tk * head_dim / 32 * 18;
            K_block_base += kv_abs * head_dim / 32 * 18;

            const uint8_t *V_block_base = V_blocks + (size_t)batch_idx * n_kv * Tk * head_dim / 32 * 18;
            V_block_base += kv_head_idx * Tk * head_dim / 32 * 18;
            V_block_base += kv_abs * head_dim / 32 * 18;

            // Compute KQ for our Q heads using dp4a
            float KQ_reg[2] = {0.0f, 0.0f};
            for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
                int *q_q8 = Q_q8 + warp_id * qh_per_warp * (HEAD_DIM / 4) + qh_idx * (HEAD_DIM / 4);
                float2 *q_ds = Q_ds + warp_id * qh_per_warp * (HEAD_DIM / 8) + qh_idx * (HEAD_DIM / 8);
                KQ_reg[qh_idx] = vec_dot_q4_0_dp4a(K_block_base, q_q8, q_ds, head_dim, lane_id);
                KQ_reg[qh_idx] *= softmax_scale;
            }

            // Online softmax + V accumulation (matching llama.cpp exactly)
            for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
                float score = KQ_reg[qh_idx];
                if (isnan(score)) score = -INFINITY;

                float m_new = fmaxf(KQ_max[qh_idx] + FATTN_KQ_MAX_OFFSET, score + FATTN_KQ_MAX_OFFSET) - FATTN_KQ_MAX_OFFSET;
                float alpha = expf(KQ_max[qh_idx] - m_new);
                float exp_score = expf(score - m_new);

                // Update online softmax state
                KQ_sum[qh_idx] = KQ_sum[qh_idx] * alpha + exp_score;
                KQ_max[qh_idx] = m_new;

                // Accumulate V (each thread handles 2 elements per head)
                for (int e = 0; e < 2; e++) {
                    int d = tid + e * (n_threads / 2);
                    if (d < head_dim) {
                        // Use Q2_0 V dequantization for TurboQuant+ asymmetric path
                        float2 v_val = dequant_q2_0_V_float2(V_block_base, d);
                        VKQ[qh_idx][e] = VKQ[qh_idx][e] * alpha + exp_score * v_val.x;
                        if (d + 1 < head_dim) {
                            VKQ[qh_idx][e] = VKQ[qh_idx][e] * alpha + exp_score * v_val.y;
                        }
                    }
                }
            }
        }
    }

    // Cross-KV-tile reduction: store per-tile KQ_max and KQ_sum
    int local_idx = warp_id * qh_per_warp;
    for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
        KQ_max_shared[local_idx + qh_idx] = KQ_max[qh_idx];
        KQ_sum_shared[local_idx + qh_idx] = KQ_sum[qh_idx];
    }
    __syncthreads();

    // Reduce across KV tiles (only warp 0 lane 0 does this)
    if (warp_id == 0 && lane_id == 0) {
        for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
            float m = -FLT_MAX/2.0f;
            float l = 0.0f;
            int idx = 0;
            for (int t = 0; t < gridDim.y; t++) {
                idx = t * n_warps * qh_per_warp + qh_idx;
                float m_t = KQ_max_shared[idx] + FATTN_KQ_MAX_OFFSET;
                float l_t = KQ_sum_shared[idx];
                float m_new = fmaxf(m, m_t);
                float alpha = expf(m - m_new);
                l = l * alpha + l_t * expf(m_t - m_new);
                m = m_new;
            }
            KQ_max_shared[idx] = m - FATTN_KQ_MAX_OFFSET;
            KQ_sum_shared[idx] = l;
        }
    }
    __syncthreads();

    // Rescale VKQ with final l = KQ_sum_final
    for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
        float l_final = KQ_sum_shared[local_idx + qh_idx];
        if (l_final > 0.0f) {
            float scale = KQ_sum[qh_idx] / l_final;
            for (int e = 0; e < 2; e++) {
                VKQ[qh_idx][e] *= scale;
            }
        }
    }
    __syncthreads();

    // Write output: each thread writes 2 elements per head
    for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
        const int qh_global = kv_head_idx * ncols + qh_base + qh_idx;
        for (int e = 0; e < 2; e++) {
            int d = tid + e * (n_threads / 2);
            if (d < head_dim) {
                int out_idx = batch_idx * n_q * Tq * head_dim + qh_global * Tq * head_dim + q_abs * head_dim + d;
                O[out_idx] = __float2half_rn(VKQ[qh_idx][e]);
            }
        }
    }
}

// Host launchers
void launch_flash_attn_q4_0_prefill(
    const half *Q, const uint8_t *K_blocks, const uint8_t *V_blocks, half *O,
    float softmax_scale, bool causal_mask, int window_size,
    int B, int Tq, int Tk, int q_tile_size,
    cudaStream_t stream
) {
    const int n_q = 16, n_kv = 2, head_dim = 256;
    const int q_heads_per_kv = 8;

    dim3 block(128);
    dim3 grid(
        (Tq + q_tile_size - 1) / q_tile_size,
        (Tk + 127) / 128,
        B * n_kv * (q_heads_per_kv / 4)
    );

    size_t q_smem = 4 * 2 * 256 * sizeof(half);
    size_t q_q8 = 4 * 2 * 64 * sizeof(int);
    size_t q_ds = 4 * 2 * 32 * sizeof(float2);
    size_t kq_max_sum = 128 * 4 * 2 * 2 * sizeof(float);
    size_t smem_bytes = q_smem + q_q8 + q_ds + kq_max_sum;

    flash_attn_q4_0_prefill_kernel<32, 256, 16, 2, 8, 128, 128><<<grid, block, smem_bytes, stream>>>(
        Q, K_blocks, V_blocks, O,
        softmax_scale, causal_mask, window_size,
        B, Tq, Tk, q_tile_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "flash_attn_q4_0_prefill launch failed: %s\n", cudaGetErrorString(err));
    }
}

// Decode path (C=1): COLS_PER_BLOCK = 1
void launch_flash_attn_q4_0_decode(
    const half *Q, const uint8_t *K_blocks, const uint8_t *V_blocks, half *O,
    float softmax_scale, bool causal_mask, int window_size,
    int B, int Tq, int Tk, int q_tile_size,
    cudaStream_t stream
) {
    const int n_q = 16, n_kv = 2, head_dim = 256;
    const int q_heads_per_kv = 8;

    dim3 block(128);
    dim3 grid(
        1,
        (Tk + 127) / 128,
        B * n_kv * (q_heads_per_kv / 4)
    );

    size_t q_smem = 4 * 2 * 256 * sizeof(half);
    size_t q_q8 = 4 * 2 * 64 * sizeof(int);
    size_t q_ds = 4 * 2 * 32 * sizeof(float2);
    size_t kq_max_sum = 1 * 4 * 2 * 2 * sizeof(float);  // Only 1 Q position
    size_t smem_bytes = q_smem + q_q8 + q_ds + kq_max_sum;

    flash_attn_q4_0_decode_kernel<256, 16, 2, 8, 128, 128><<<grid, block, smem_bytes, stream>>>(
        Q, K_blocks, V_blocks, O,
        softmax_scale, causal_mask, window_size,
        B, Tq, Tk, 1
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "flash_attn_q4_0_decode launch failed: %s\n", cudaGetErrorString(err));
    }
}

// ================================================================
// PagedAttention Kernels (vLLM Style)
// ================================================================

/**
 * PagedAttention Prefill Kernel with Q4_0
 * 
 * Adds one level of indirection: block_table -> physical blocks
 */
template<
    int COLS_PER_BLOCK,
    int HEAD_DIM,
    int N_Q_HEADS,
    int N_KV_HEADS,
    int Q_HEADS_PER_KV,
    int BLOCK_THREADS,
    int KV_TILE_SIZE
>
__global__ void flash_attn_q4_0_paged_prefill_kernel(
    const half *__restrict__ Q,           // [B, Hq, Tq, HEAD_DIM] FP16
    const int *__restrict__ block_table,  // [B * max_blocks] physical block IDs
    const uint8_t *__restrict__ K_pool,   // All Q4_0 blocks contiguous
    const uint8_t *__restrict__ V_pool,   // All Q4_0 blocks contiguous
    half *__restrict__ O,                 // [B, Hq, Tq, HEAD_DIM] FP16
    const float softmax_scale,
    const bool causal_mask,
    const int window_size,
    const int B,
    const int Tq,
    const int Tk,
    const int q_tile_size
) {
    const int head_dim = HEAD_DIM;
    const int n_q = N_Q_HEADS;
    const int n_kv = N_KV_HEADS;
    const int ncols = Q_HEADS_PER_KV;  // 8
    const int warp_size = 32;
    const int n_threads = BLOCK_THREADS;  // 128
    const int n_warps = n_threads / warp_size;  // 4

    const int tid = threadIdx.x;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;

    // Grid mapping:
    // blockIdx.x = q_tile_idx
    // blockIdx.y = kv_tile_idx (parallel KV chunks)
    // blockIdx.z = batch_idx * n_kv * (ncols/n_warps) + kv_head_idx * (ncols/n_warps) + qh_group
    const int q_tile_idx = blockIdx.x;
    const int kv_tile_idx = blockIdx.y;
    const int batch_kv_qh = blockIdx.z;

    const int qh_groups = ncols / n_warps;  // 8/4 = 2
    const int batch_idx = batch_kv_qh / (n_kv * qh_groups);
    const int residual = batch_kv_qh % (n_kv * qh_groups);
    const int kv_head_idx = residual / qh_groups;
    const int qh_group = residual % qh_groups;  // 0..1

    if (q_tile_idx * q_tile_size >= Tq) return;
    if (kv_head_idx >= n_kv) return;
    if (batch_idx >= B) return;

    const int q_start = q_tile_idx * q_tile_size;
    const int q_end = min(q_start + q_tile_size, Tq);
    const int q_len = q_end - q_start;

    const int kv_start = kv_tile_idx * KV_TILE_SIZE;
    const int kv_end = min(kv_start + KV_TILE_SIZE, Tk);
    const int kv_len = kv_end - kv_start;

    if (kv_len <= 0) return;

    // Each warp handles qh_per_warp = ncols/n_warps = 2 Q heads
    const int qh_per_warp = ncols / n_warps;
    const int qh_base = qh_group * n_warps + warp_id * qh_per_warp;
    if (qh_base >= ncols) return;

    // Shared memory allocation (same as non-paged + block_table)
    extern __shared__ char smem[];
    size_t offset = 0;
    half *Q_smem = (half*)(smem + offset);
    offset += n_warps * qh_per_warp * HEAD_DIM * sizeof(half);
    int *Q_q8 = (int*)(smem + offset);
    offset += n_warps * qh_per_warp * (HEAD_DIM / 4) * sizeof(int);
    float2 *Q_ds = (float2*)(smem + offset);
    offset += n_warps * qh_per_warp * (HEAD_DIM / 8) * sizeof(float2);
    float *KQ_max_shared = (float*)(smem + offset);
    offset += q_len * n_warps * qh_per_warp * sizeof(float);
    float *KQ_sum_shared = (float*)(smem + offset);

    // Load Q into shared memory for our Q heads
    for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
        const int qh_global = kv_head_idx * ncols + qh_base + qh_idx;
        for (int d = tid; d < head_dim; d += n_threads) {
            int src = batch_idx * n_q * Tq * head_dim + qh_global * Tq * head_dim + q_start * head_dim + d;
            Q_smem[warp_id * qh_per_warp * HEAD_DIM + qh_idx * HEAD_DIM + d] = Q[src];
        }
    }
    __syncthreads();

    // Quantize Q from shared memory to q8_1 for our Q heads
    for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
        half *Q_reg = Q_smem + warp_id * qh_per_warp * HEAD_DIM + qh_idx * HEAD_DIM;
        int *q_q8 = Q_q8 + warp_id * qh_per_warp * (HEAD_DIM / 4) + qh_idx * (HEAD_DIM / 4);
        float2 *q_ds = Q_ds + warp_id * qh_per_warp * (HEAD_DIM / 8) + qh_idx * (HEAD_DIM / 8);
        quantize_Q_to_q8_1(Q_reg, q_q8, q_ds, head_dim, lane_id);
    }
    __syncthreads();

    // Process each Q position in this tile
    for (int q_rel = 0; q_rel < q_len; q_rel++) {
        int q_abs = q_start + q_rel;

        // Per-Q-position softmax state
        float KQ_max[2] = {-FLT_MAX/2.0f, -FLT_MAX/2.0f};
        float KQ_sum[2] = {0.0f, 0.0f};
        float VKQ[2][2] = {{0.0f, 0.0f}};

        // Process this KV tile for this Q position
        for (int kv = 0; kv < kv_len; kv++) {
            int kv_abs = kv_start + kv;

            bool masked = false;
            if (causal_mask && kv_abs > q_abs) masked = true;
            if (window_size > 0 && kv_abs < q_abs - window_size) masked = true;

            if (!masked) {
                // PagedAttention: lookup physical block for this KV position
                int block_id = block_table[batch_idx * gridDim.x * gridDim.y * gridDim.z + 
                              kv_abs / KV_BLOCK_SIZE];
                const uint8_t *K_block_base = K_pool + (size_t)block_id * KV_BLOCK_SIZE * head_dim / 32 * 18;
                K_block_base += (kv_abs % KV_BLOCK_SIZE) * head_dim / 32 * 18;

                const uint8_t *V_block_base = V_pool + (size_t)block_id * KV_BLOCK_SIZE * head_dim / 32 * 18;
                V_block_base += (kv_abs % KV_BLOCK_SIZE) * head_dim / 32 * 18;

                // Compute KQ for our Q heads using dp4a
                float KQ_reg[2] = {0.0f, 0.0f};
                for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
                    int *q_q8 = Q_q8 + warp_id * qh_per_warp * (HEAD_DIM / 4) + qh_idx * (HEAD_DIM / 4);
                    float2 *q_ds = Q_ds + warp_id * qh_per_warp * (HEAD_DIM / 8) + qh_idx * (HEAD_DIM / 8);
                    KQ_reg[qh_idx] = vec_dot_q4_0_dp4a(K_block_base, q_q8, q_ds, head_dim, lane_id);
                    KQ_reg[qh_idx] *= softmax_scale;
                }

                // Online softmax + V accumulation (matching llama.cpp exactly)
                for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
                    float score = KQ_reg[qh_idx];
                    if (isnan(score)) score = -INFINITY;

                    float m_new = fmaxf(KQ_max[qh_idx] + FATTN_KQ_MAX_OFFSET, score + FATTN_KQ_MAX_OFFSET) - FATTN_KQ_MAX_OFFSET;
                    float alpha = expf(KQ_max[qh_idx] - m_new);
                    float exp_score = expf(score - m_new);

                    KQ_sum[qh_idx] = KQ_sum[qh_idx] * alpha + exp_score;
                    KQ_max[qh_idx] = m_new;

                    // Accumulate V
                    for (int e = 0; e < 2; e++) {
                        int d = tid + e * (n_threads / 2);
                        if (d < head_dim) {
                            float2 v_val = dequant_q4_0_V_float2(V_block_base, d);
                            VKQ[qh_idx][e] = VKQ[qh_idx][e] * alpha + exp_score * v_val.x;
                            if (d + 1 < head_dim) {
                                VKQ[qh_idx][e] = VKQ[qh_idx][e] * alpha + exp_score * v_val.y;
                            }
                        }
                    }
                }
            }
        }

        // Cross-KV-tile reduction
        int local_idx = q_rel * n_warps * qh_per_warp + warp_id * qh_per_warp;
        for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
            KQ_max_shared[local_idx + qh_idx] = KQ_max[qh_idx];
            KQ_sum_shared[local_idx + qh_idx] = KQ_sum[qh_idx];
        }
        __syncthreads();

        if (warp_id == 0 && lane_id == 0) {
            for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
                float m = -FLT_MAX/2.0f;
                float l = 0.0f;
                for (int t = 0; t < gridDim.y; t++) {
                    int idx = t * q_len * n_warps * qh_per_warp + q_rel * n_warps * qh_per_warp + qh_idx;
                    float m_t = KQ_max_shared[idx] + FATTN_KQ_MAX_OFFSET;
                    float l_t = KQ_sum_shared[idx];
                    float m_new = fmaxf(m, m_t);
                    float alpha = expf(m - m_new);
                    l = l * alpha + l_t * expf(m_t - m_new);
                    m = m_new;
                }
                KQ_max_shared[local_idx + qh_idx] = m - FATTN_KQ_MAX_OFFSET;
                KQ_sum_shared[local_idx + qh_idx] = l;
            }
        }
        __syncthreads();

        // Rescale VKQ with final l
        for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
            float l_final = KQ_sum_shared[local_idx + qh_idx];
            if (l_final > 0.0f) {
                float scale = KQ_sum[qh_idx] / l_final;
                for (int e = 0; e < 2; e++) {
                    VKQ[qh_idx][e] *= scale;
                }
            }
        }
        __syncthreads();

        // Write output
        for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
            const int qh_global = kv_head_idx * ncols + qh_base + qh_idx;
            for (int e = 0; e < 2; e++) {
                int d = tid + e * (n_threads / 2);
                if (d < head_dim) {
                    int out_idx = batch_idx * n_q * Tq * head_dim + qh_global * Tq * head_dim + q_abs * head_dim + d;
                    O[out_idx] = __float2half_rn(VKQ[qh_idx][e]);
                }
            }
        }
    }
}

/**
 * PagedAttention Decode Kernel with Q4_0
 */
template<
    int HEAD_DIM,
    int N_Q_HEADS,
    int N_KV_HEADS,
    int Q_HEADS_PER_KV,
    int BLOCK_THREADS,
    int KV_TILE_SIZE
>
__global__ void flash_attn_q4_0_paged_decode_kernel(
    const half *__restrict__ Q,           // [B, Hq, Tq, HEAD_DIM] FP16, Tq=1
    const int *__restrict__ block_table,  // [B * max_blocks] physical block IDs
    const uint8_t *__restrict__ K_pool,   // All Q4_0 blocks contiguous
    const uint8_t *__restrict__ V_pool,   // All Q4_0 blocks contiguous
    half *__restrict__ O,                 // [B, Hq, Tq, HEAD_DIM] FP16
    const float softmax_scale,
    const bool causal_mask,
    const int window_size,
    const int B,
    const int Tq,
    const int Tk,
    const int q_tile_size
) {
    const int head_dim = HEAD_DIM;
    const int n_q = N_Q_HEADS;
    const int n_kv = N_KV_HEADS;
    const int ncols = Q_HEADS_PER_KV;  // 8
    const int warp_size = 32;
    const int n_threads = BLOCK_THREADS;  // 128
    const int n_warps = n_threads / warp_size;  // 4

    const int tid = threadIdx.x;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;

    // Grid mapping:
    // blockIdx.x = 0 (only 1 Q tile for decode)
    // blockIdx.y = kv_tile_idx (parallel KV chunks)
    // blockIdx.z = batch_idx * n_kv * (ncols/n_warps) + kv_head_idx * (ncols/n_warps) + qh_group
    const int kv_tile_idx = blockIdx.y;
    const int batch_kv_qh = blockIdx.z;

    const int qh_groups = ncols / n_warps;  // 8/4 = 2
    const int batch_idx = batch_kv_qh / (n_kv * qh_groups);
    const int residual = batch_kv_qh % (n_kv * qh_groups);
    const int kv_head_idx = residual / qh_groups;
    const int qh_group = residual % qh_groups;  // 0..1

    if (kv_head_idx >= n_kv) return;
    if (batch_idx >= B) return;

    const int kv_start = kv_tile_idx * KV_TILE_SIZE;
    const int kv_end = min(kv_start + KV_TILE_SIZE, Tk);
    const int kv_len = kv_end - kv_start;

    if (kv_len <= 0) return;

    // Each warp handles qh_per_warp = ncols/n_warps = 2 Q heads
    const int qh_per_warp = ncols / n_warps;
    const int qh_base = qh_group * n_warps + warp_id * qh_per_warp;
    if (qh_base >= ncols) return;

    // Shared memory allocation
    extern __shared__ char smem[];
    size_t offset = 0;
    half *Q_smem = (half*)(smem + offset);
    offset += n_warps * qh_per_warp * HEAD_DIM * sizeof(half);
    int *Q_q8 = (int*)(smem + offset);
    offset += n_warps * qh_per_warp * (HEAD_DIM / 4) * sizeof(int);
    float2 *Q_ds = (float2*)(smem + offset);
    offset += n_warps * qh_per_warp * (HEAD_DIM / 8) * sizeof(float2);
    float *KQ_max_shared = (float*)(smem + offset);
    offset += n_warps * qh_per_warp * sizeof(float);
    float *KQ_sum_shared = (float*)(smem + offset);

    // Load Q into shared memory (C=1, q_start=0)
    for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
        const int qh_global = kv_head_idx * ncols + qh_base + qh_idx;
        for (int d = tid; d < head_dim; d += n_threads) {
            int src = batch_idx * n_q * Tq * head_dim + qh_global * Tq * head_dim + 0 * head_dim + d;
            Q_smem[warp_id * qh_per_warp * HEAD_DIM + qh_idx * HEAD_DIM + d] = Q[src];
        }
    }
    __syncthreads();

    // Quantize Q to q8_1
    for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
        half *Q_reg = Q_smem + warp_id * qh_per_warp * HEAD_DIM + qh_idx * HEAD_DIM;
        int *q_q8 = Q_q8 + warp_id * qh_per_warp * (HEAD_DIM / 4) + qh_idx * (HEAD_DIM / 4);
        float2 *q_ds = Q_ds + warp_id * qh_per_warp * (HEAD_DIM / 8) + qh_idx * (HEAD_DIM / 8);
        quantize_Q_to_q8_1(Q_reg, q_q8, q_ds, head_dim, lane_id);
    }
    __syncthreads();

    // Single Q position (C=1, q_rel=0)
    int q_abs = 0;

    // Per-Q-position softmax state
    float KQ_max[2] = {-FLT_MAX/2.0f, -FLT_MAX/2.0f};
    float KQ_sum[2] = {0.0f, 0.0f};
    float VKQ[2][2] = {{0.0f, 0.0f}};

    // First pass: Process KV tiles, compute KQ, online softmax, accumulate VKQ
    for (int kv = 0; kv < kv_len; kv++) {
        int kv_abs = kv_start + kv;

        bool masked = false;
        if (causal_mask && kv_abs > q_abs) masked = true;
        if (window_size > 0 && kv_abs < q_abs - window_size) masked = true;

        if (!masked) {
            // PagedAttention: lookup physical block
            int block_id = block_table[batch_idx * gridDim.x * gridDim.y * gridDim.z + 
                          kv_abs / KV_BLOCK_SIZE];
            const uint8_t *K_block_base = K_pool + (size_t)block_id * KV_BLOCK_SIZE * head_dim / 32 * 18;
            K_block_base += (kv_abs % KV_BLOCK_SIZE) * head_dim / 32 * 18;

            const uint8_t *V_block_base = V_pool + (size_t)block_id * KV_BLOCK_SIZE * head_dim / 32 * 18;
            V_block_base += (kv_abs % KV_BLOCK_SIZE) * head_dim / 32 * 18;

            // Compute KQ using dp4a
            float KQ_reg[2] = {0.0f, 0.0f};
            for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
                int *q_q8 = Q_q8 + warp_id * qh_per_warp * (HEAD_DIM / 4) + qh_idx * (HEAD_DIM / 4);
                float2 *q_ds = Q_ds + warp_id * qh_per_warp * (HEAD_DIM / 8) + qh_idx * (HEAD_DIM / 8);
                KQ_reg[qh_idx] = vec_dot_q4_0_dp4a(K_block_base, q_q8, q_ds, head_dim, lane_id);
                KQ_reg[qh_idx] *= softmax_scale;
            }

            // Online softmax + V accumulation
            for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
                float score = KQ_reg[qh_idx];
                if (isnan(score)) score = -INFINITY;

                float m_new = fmaxf(KQ_max[qh_idx] + FATTN_KQ_MAX_OFFSET, score + FATTN_KQ_MAX_OFFSET) - FATTN_KQ_MAX_OFFSET;
                float alpha = expf(KQ_max[qh_idx] - m_new);
                float exp_score = expf(score - m_new);

                KQ_sum[qh_idx] = KQ_sum[qh_idx] * alpha + exp_score;
                KQ_max[qh_idx] = m_new;

                // Accumulate V
                for (int e = 0; e < 2; e++) {
                    int d = tid + e * (n_threads / 2);
                    if (d < head_dim) {
                        float2 v_val = dequant_q4_0_V_float2(V_block_base, d);
                        VKQ[qh_idx][e] = VKQ[qh_idx][e] * alpha + exp_score * v_val.x;
                        if (d + 1 < head_dim) {
                            VKQ[qh_idx][e] = VKQ[qh_idx][e] * alpha + exp_score * v_val.y;
                        }
                    }
                }
            }
        }
    }

    // Cross-KV-tile reduction
    int local_idx = warp_id * qh_per_warp;
    for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
        KQ_max_shared[local_idx + qh_idx] = KQ_max[qh_idx];
        KQ_sum_shared[local_idx + qh_idx] = KQ_sum[qh_idx];
    }
    __syncthreads();

    if (warp_id == 0 && lane_id == 0) {
        for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
            float m = -FLT_MAX/2.0f;
            float l = 0.0f;
            int idx = 0;
            for (int t = 0; t < gridDim.y; t++) {
                idx = t * n_warps * qh_per_warp + qh_idx;
                float m_t = KQ_max_shared[idx] + FATTN_KQ_MAX_OFFSET;
                float l_t = KQ_sum_shared[idx];
                float m_new = fmaxf(m, m_t);
                float alpha = expf(m - m_new);
                l = l * alpha + l_t * expf(m_t - m_new);
                m = m_new;
            }
            KQ_max_shared[idx] = m - FATTN_KQ_MAX_OFFSET;
            KQ_sum_shared[idx] = l;
        }
    }
    __syncthreads();

    // Rescale VKQ with final l
    for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
        float l_final = KQ_sum_shared[local_idx + qh_idx];
        if (l_final > 0.0f) {
            float scale = KQ_sum[qh_idx] / l_final;
            for (int e = 0; e < 2; e++) {
                VKQ[qh_idx][e] *= scale;
            }
        }
    }
    __syncthreads();

    // Write output
    for (int qh_idx = 0; qh_idx < qh_per_warp; qh_idx++) {
        const int qh_global = kv_head_idx * ncols + qh_base + qh_idx;
        for (int e = 0; e < 2; e++) {
            int d = tid + e * (n_threads / 2);
            if (d < head_dim) {
                int out_idx = batch_idx * n_q * Tq * head_dim + qh_global * Tq * head_dim + q_abs * head_dim + d;
                O[out_idx] = __float2half_rn(VKQ[qh_idx][e]);
            }
        }
    }
}

// Host launchers for paged attention
void launch_flash_attn_q4_0_paged_prefill(
    const half *Q, const int *block_table, const uint8_t *K_pool, const uint8_t *V_pool, half *O,
    float softmax_scale, bool causal_mask, int window_size,
    int B, int Tq, int Tk, int q_tile_size,
    cudaStream_t stream
) {
    const int n_q = 16, n_kv = 2, head_dim = 256;
    const int q_heads_per_kv = 8;

    dim3 block(128);
    dim3 grid(
        (Tq + q_tile_size - 1) / q_tile_size,
        (Tk + 127) / 128,
        B * n_kv * (q_heads_per_kv / 4)
    );

    size_t q_smem = 4 * 2 * 256 * sizeof(half);
    size_t q_q8 = 4 * 2 * 64 * sizeof(int);
    size_t q_ds = 4 * 2 * 32 * sizeof(float2);
    size_t kq_max_sum = 128 * 4 * 2 * 2 * sizeof(float);
    size_t smem_bytes = q_smem + q_q8 + q_ds + kq_max_sum;

    flash_attn_q4_0_paged_prefill_kernel<32, 256, 16, 2, 8, 128, 128><<<grid, block, smem_bytes, stream>>>(
        Q, block_table, K_pool, V_pool, O,
        softmax_scale, causal_mask, window_size,
        B, Tq, Tk, q_tile_size
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "flash_attn_q4_0_paged_prefill launch failed: %s\n", cudaGetErrorString(err));
    }
}

void launch_flash_attn_q4_0_paged_decode(
    const half *Q, const int *block_table, const uint8_t *K_pool, const uint8_t *V_pool, half *O,
    float softmax_scale, bool causal_mask, int window_size,
    int B, int Tq, int Tk, int q_tile_size,
    cudaStream_t stream
) {
    const int n_q = 16, n_kv = 2, head_dim = 256;
    const int q_heads_per_kv = 8;

    dim3 block(128);
    dim3 grid(
        1,
        (Tk + 127) / 128,
        B * n_kv * (q_heads_per_kv / 4)
    );

    size_t q_smem = 4 * 2 * 256 * sizeof(half);
    size_t q_q8 = 4 * 2 * 64 * sizeof(int);
    size_t q_ds = 4 * 2 * 32 * sizeof(float2);
    size_t kq_max_sum = 1 * 4 * 2 * 2 * sizeof(float);
    size_t smem_bytes = q_smem + q_q8 + q_ds + kq_max_sum;

    flash_attn_q4_0_paged_decode_kernel<256, 16, 2, 8, 128, 128><<<grid, block, smem_bytes, stream>>>(
        Q, block_table, K_pool, V_pool, O,
        softmax_scale, causal_mask, window_size,
        B, Tq, Tk, 1
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "flash_attn_q4_0_paged_decode launch failed: %s\n", cudaGetErrorString(err));
    }
}