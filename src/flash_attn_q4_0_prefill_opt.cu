// Optimized Flash Attention Q4_0 Prefill Kernel
// Same pattern as decode_opt but handles multiple Q tokens per block
// No KV grid-stride, no cross-tile reduction - single block loops all KV
// Grid: (B * n_kv, n_q_tiles)  Block: (32, 4) = 128 threads

#include "flash_attn_q4_0_opt.cuh"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include <cmath>
#include <cfloat>
#include <cstdio>

#define WARP_SIZE 32
#define WARP_ROWS 4
#define QK4_0 32
#define QI8_1 32
#define FATTN_KQ_MAX_OFFSET 2.0794415416798357f  // 3 * ln(2)
#define QK_PER_Q8_GROUP 8  // QI8_1 / 4 = 32/4 = 8 int32 per Q_ds group

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float tmp = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, tmp);
    }
    return val;
}

__device__ __forceinline__ half2 load_half2(const half* addr) {
    return *reinterpret_cast<const half2*>(addr);
}

__device__ __forceinline__ void store_half2(half* addr, half2 val) {
    *reinterpret_cast<half2*>(addr) = val;
}

__device__ static inline void quantize_q_f32_to_q8_1(const float* src, int* dst_i32, float2* dst_ds, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = fabsf(isnan(src[i]) ? 0.0f : src[i]);
        if (v > max_abs) max_abs = v;
    }
    float q_scale = max_abs / 127.0f;
    if (q_scale == 0.0f) q_scale = 1.0f;

    for (int i = 0; i < n / QI8_1; i++) {
        int32_t sum = 0;
        int32_t isum = 0;
        for (int j = 0; j < QI8_1; j++) {
            float v = isnan(src[i * QI8_1 + j]) ? 0.0f : src[i * QI8_1 + j];
            int8_t q = __float2int_rn(v / q_scale);
            sum |= (q & 0xFF) << (8 * j);
            isum += q;
        }
        dst_i32[i] = sum;
        dst_ds[i] = make_float2(q_scale, isum * q_scale / QI8_1);
    }
}

__device__ __forceinline__ float vec_dot_q4_0_dp4a(const uint8_t* K_block, const int* Q_q8, const float2* Q_ds, int head_dim, int lane_id) {
    const uint8_t* K_ptr = K_block;
    float sum = 0.0f;

    for (int k_KQ_0 = lane_id; k_KQ_0 < head_dim / 4; k_KQ_0 += WARP_SIZE) {
        const int ib    = (k_KQ_0 * 4) / QK4_0;
        const int iqs4  = (k_KQ_0 * 4) % (QK4_0/2);
        const int shift = ((k_KQ_0 * 4) % QK4_0) / (QK4_0/2);

        size_t block_offset = ib * 18;
        uint16_t d_raw = K_ptr[block_offset] | (K_ptr[block_offset + 1] << 8);
        float k_scale = __half2float(*reinterpret_cast<const __half*>(&d_raw));

        const uint8_t* qs_ptr = K_ptr + block_offset + 2 + 4 * iqs4;
        int v = qs_ptr[0] | (qs_ptr[1] << 8) | (qs_ptr[2] << 16) | (qs_ptr[3] << 24);
        v = (v >> (4 * shift)) & 0x0F0F0F0F;

        int k_int8 = v - 0x08080808;

        int u = Q_q8[k_KQ_0];
        int sumi = __dp4a(u, k_int8, 0);

        const float2 Q_ds_val = Q_ds[k_KQ_0 / QK_PER_Q8_GROUP];

        sum += k_scale * (sumi * Q_ds_val.x - 4.0f * Q_ds_val.y);
    }

    return warp_reduce_sum(sum);
}

__device__ __forceinline__ float2 dequant_q4_0_V(const uint8_t* V_block, int d) {
    const uint8_t* V_ptr = V_block;
    int bk = d / 32;
    int el = d % 32;

    size_t block_offset = bk * 18;
    uint16_t d_raw = V_ptr[block_offset] | (V_ptr[block_offset + 1] << 8);
    float scale = __half2float(*reinterpret_cast<const __half*>(&d_raw));

    const uint8_t* vp = V_ptr + block_offset + 2;
    int q0 = (vp[el/2] >> (4 * (el % 2))) & 0xF;
    int q1 = (vp[(el+1)/2] >> (4 * ((el+1) % 2))) & 0xF;
    q0 -= 8; q1 -= 8;

    return make_float2(scale * q0, scale * q1);
}

// ================================================================
// Optimized Prefill Kernel - single block processes all KV for its Q tile
// ================================================================
__global__ void flash_attn_q4_0_prefill_opt_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K_pool,
    const uint8_t* __restrict__ V_pool,
    half* __restrict__ O,
    float softmax_scale,
    bool causal_mask,
    int window_size,
    int B, int Tq, int Tk,
    int q_tile_size,
    int n_q, int n_kv, int q_heads_per_kv, int head_dim, int kv_dim
) {
    const int qh_per_kv = q_heads_per_kv;
    const int qh_per_warp = qh_per_kv / WARP_ROWS;
    const int heads_per_warp = qh_per_warp;

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;

    const int linear_kv = blockIdx.x;
    const int batch = linear_kv / n_kv;
    const int kv_head = linear_kv % n_kv;

    const int q_tile_idx = blockIdx.y;
    const int q_start = q_tile_idx * q_tile_size;
    if (q_start >= Tq) return;
    const int q_end = min(q_start + q_tile_size, Tq);
    const int q_len = q_end - q_start;

    if (batch >= B) return;
    if (kv_head >= n_kv) return;

    const int q_head_start = kv_head * qh_per_kv;

    extern __shared__ char smem[];
    size_t offset = 0;
    half* Q_smem = reinterpret_cast<half*>(smem + offset);
    offset += q_len * WARP_ROWS * qh_per_warp * head_dim * sizeof(half);
    int* Q_q8 = reinterpret_cast<int*>(smem + offset);
    offset += WARP_ROWS * qh_per_kv * (head_dim / 4) * sizeof(int);
    float2* Q_ds = reinterpret_cast<float2*>(smem + offset);
    offset += WARP_ROWS * qh_per_kv * (head_dim / QI8_1) * sizeof(float2);

    // Load Q for all q_len positions in this tile
    for (int q_rel = 0; q_rel < q_len; q_rel++) {
        int q_abs = q_start + q_rel;
        for (int qh_rel = 0; qh_rel < qh_per_kv; qh_rel++) {
            int qh_abs = q_head_start + qh_rel;
            if (qh_abs >= n_q) continue;
            int warp_rel = qh_rel / qh_per_warp;
            int head_in_warp = qh_rel % qh_per_warp;
            half* dst = Q_smem + q_rel * WARP_ROWS * qh_per_warp * head_dim + warp_rel * qh_per_warp * head_dim + head_in_warp * head_dim;
            const half* src = Q + batch * n_q * Tq * head_dim + qh_abs * Tq * head_dim + q_abs * head_dim;
            for (int d = (warp_id * WARP_SIZE + lane_id); d < head_dim; d += blockDim.x * blockDim.y) {
                dst[d] = src[d];
            }
        }
    }
    __syncthreads();

    // Quantize Q for all q_len positions - each warp handles its 2 heads for all q_rel
    for (int q_rel = 0; q_rel < q_len; q_rel++) {
        for (int qh_rel = warp_id * qh_per_warp; qh_rel < (warp_id + 1) * qh_per_warp && qh_rel < qh_per_kv; qh_rel++) {
            int qh_abs = q_head_start + qh_rel;
            if (qh_abs >= n_q) continue;
            half* Q_reg = Q_smem + q_rel * WARP_ROWS * qh_per_warp * head_dim + warp_id * qh_per_warp * head_dim + (qh_rel % qh_per_warp) * head_dim;
            int* q_q8 = Q_q8 + warp_id * qh_per_kv * (head_dim / 4) + qh_rel * (head_dim / 4);
            float2* q_ds = Q_ds + warp_id * qh_per_kv * (head_dim / QI8_1) + qh_rel * (head_dim / QI8_1);

            if (lane_id == 0) {
                float Q_f32[256];
                for (int d = 0; d < head_dim; d += 2) {
                    half2 v = load_half2(Q_reg + d);
                    float x = __half2float(v.x);
                    float y = __half2float(v.y);
                    // Sanitize NaN in Q
                    Q_f32[d] = (isnan(x) ? 0.0f : x) * softmax_scale;
                    Q_f32[d+1] = (isnan(y) ? 0.0f : y) * softmax_scale;
                }
                quantize_q_f32_to_q8_1(Q_f32, q_q8, q_ds, head_dim);
            }
        }
    }
    __syncthreads();

    // Process each Q position sequentially
    for (int q_rel = 0; q_rel < q_len; q_rel++) {
        int q_abs = q_start + q_rel;

        float m_reg[2] = {-FLT_MAX/2.0f, -FLT_MAX/2.0f};
        float l_reg[2] = {0.0f, 0.0f};
        float2 VKQ_reg[2][8];
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            #pragma unroll
            for (int e = 0; e < 8; e++) VKQ_reg[hh][e] = make_float2(0.0f, 0.0f);
        }

        // Loop over ALL KV tokens (no grid-stride, no cross-block reduction)
        for (int kv = 0; kv < Tk; kv++) {
            bool masked = false;
            if (causal_mask && kv > q_abs) masked = true;
            if (window_size > 0 && kv < q_abs - window_size) masked = true;
            if (masked) continue;

            const int kv_blocks_per_token = kv_dim / 32;
            const int head_blocks = head_dim / 32;
            const int block_base = kv * kv_blocks_per_token + kv_head * head_blocks;
            const uint8_t* K_token = K_pool + (size_t)block_base * 18;
            const uint8_t* V_token = V_pool + (size_t)block_base * 18;

            for (int hh = 0; hh < qh_per_warp; hh++) {
                int qh_rel = warp_id * qh_per_warp + hh;
                if (qh_rel >= qh_per_kv) continue;

                int* q_q8 = Q_q8 + warp_id * qh_per_kv * (head_dim / 4) + qh_rel * (head_dim / 4);
                float2* q_ds = Q_ds + warp_id * qh_per_kv * (head_dim / QI8_1) + qh_rel * (head_dim / QI8_1);

                float score = vec_dot_q4_0_dp4a(K_token, q_q8, q_ds, head_dim, lane_id);
                if (isnan(score)) score = -FLT_MAX/2.0f;

                // Use llama.cpp exact FATTN_KQ_MAX_OFFSET formula
                float m_new = fmaxf(m_reg[hh] + FATTN_KQ_MAX_OFFSET, score + FATTN_KQ_MAX_OFFSET) - FATTN_KQ_MAX_OFFSET;
                float alpha = expf(m_reg[hh] - m_new);
                float exp_score = expf(score - m_new);

                // Clamp alpha to avoid overflow/underflow
                if (alpha > 1e30f) alpha = 1e30f;
                if (exp_score > 1e30f) exp_score = 1e30f;

                l_reg[hh] = l_reg[hh] * alpha + exp_score;
                m_reg[hh] = m_new;

                for (int d = lane_id * 2; d < head_dim; d += WARP_SIZE * 2) {
                    float2 v = dequant_q4_0_V(V_token, d);
                    if (isnan(v.x)) v.x = 0.0f;
                    if (isnan(v.y)) v.y = 0.0f;
                    int e = d / 64;
                    VKQ_reg[hh][e].x = VKQ_reg[hh][e].x * alpha + exp_score * v.x;
                    VKQ_reg[hh][e].y = VKQ_reg[hh][e].y * alpha + exp_score * v.y;
                }
            }
        }

        // Write output for this q_rel (no cross-warp reduction needed - each warp writes its 2 heads)
        for (int hh = 0; hh < qh_per_warp; hh++) {
            int qh_rel = warp_id * qh_per_warp + hh;
            if (qh_rel >= qh_per_kv) continue;
            int qh_abs = q_head_start + qh_rel;

            half* O_global = O + batch * n_q * Tq * head_dim + qh_abs * Tq * head_dim + q_abs * head_dim;
            float denom = l_reg[hh] > 0.0f ? l_reg[hh] : 1.0f;
            for (int d = lane_id * 2; d < head_dim; d += WARP_SIZE * 2) {
                int e = d / 64;
                float out_x = VKQ_reg[hh][e].x / denom;
                float out_y = VKQ_reg[hh][e].y / denom;
                if (isnan(out_x)) out_x = 0.0f;
                if (isnan(out_y)) out_y = 0.0f;
                half2 val = make_half2(__float2half_rn(out_x), __float2half_rn(out_y));
                store_half2(O_global + d, val);
            }
        }
    }
}

void launch_flash_attn_q4_0_prefill_opt(
    const half *Q, const uint8_t *K_pool, const uint8_t *V_pool, half *O,
    float softmax_scale, bool causal_mask, int window_size,
    int B, int Tq, int Tk, int q_tile_size,
    cudaStream_t stream
) {
    const int n_q = 16, n_kv = 2, head_dim = 256;
    const int q_heads_per_kv = 8;
    const int kv_dim = n_kv * head_dim;

    dim3 block(32, 4);
    dim3 grid(B * n_kv, (Tq + q_tile_size - 1) / q_tile_size);

    size_t smem_bytes = 
        q_tile_size * WARP_ROWS * 2 * head_dim * sizeof(half) +
        WARP_ROWS * q_heads_per_kv * (head_dim / 4) * sizeof(int) +
        WARP_ROWS * q_heads_per_kv * (head_dim / QI8_1) * sizeof(float2);

    if (smem_bytes > 48 * 1024) {
        fprintf(stderr, "ERROR: Prefill opt shared memory %zu KB exceeds 48KB limit\n", smem_bytes / 1024);
        return;
    }

    flash_attn_q4_0_prefill_opt_kernel<<<grid, block, smem_bytes, stream>>>(
        Q, K_pool, V_pool, O,
        softmax_scale, causal_mask, window_size,
        B, Tq, Tk, q_tile_size,
        n_q, n_kv, q_heads_per_kv, head_dim, kv_dim
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "flash_attn_q4_0_prefill_opt launch failed: %s\n", cudaGetErrorString(err));
    }
}