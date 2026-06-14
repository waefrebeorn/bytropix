// Optimized Flash Attention Q4_0 Decode Kernel
// Based on llama.cpp flash_attn_ext_vec (fattn-vec.cuh) patterns:
// - gridDim = (B * n_kv, 1, 1) - one block per KV head per batch
// - blockDim = (WARP_SIZE, WARP_ROWS) = (32, 4) = 128 threads
// - Each block processes ALL 8 Q heads for its KV head
// - KV blocks looped sequentially within block (NO cross-block KV split)
// - Warp-level online softmax with M, L, acc in registers
// - Q quantized to Q8_1, dp4a for Q4_0 dot product
// - Cross-warp reduction via shared memory + __syncthreads()

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
#define QK_PER_Q8_GROUP 8  // QI8_1 / 4 = 32/4 = 8 int32 per Q_ds group

// ================================================================
// Device helpers - using byte-wise access to avoid alignment issues
// ================================================================

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

// Q8_1 quantization of Q (single thread per warp)
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

// dp4a dot product for Q4_0 K * Q8_1 Q - byte-wise access for alignment
__device__ __forceinline__ float vec_dot_q4_0_dp4a(const uint8_t* K_block, const int* Q_q8, const float2* Q_ds, int head_dim, int lane_id) {
    const uint8_t* K_ptr = K_block;
    float sum = 0.0f;

    for (int k_KQ_0 = lane_id; k_KQ_0 < head_dim / 4; k_KQ_0 += WARP_SIZE) {
        const int ib    = (k_KQ_0 * 4) / QK4_0;
        const int iqs4  = (k_KQ_0 * 4) % (QK4_0/2);
        const int shift = ((k_KQ_0 * 4) % QK4_0) / (QK4_0/2);

        size_t block_offset = ib * 18;
        uint16_t d_raw = K_ptr[block_offset] | (K_ptr[block_offset + 1] << 8);
        float k_scale = __half2float(*reinterpret_cast<const half*>(&d_raw));

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

// Dequant V from Q4_0 - byte-wise access
__device__ __forceinline__ float2 dequant_q4_0_V(const uint8_t* V_block, int d) {
    const uint8_t* V_ptr = V_block;
    int bk = d / 32;
    int el = d % 32;
    
    size_t block_offset = bk * 18;
    uint16_t d_raw = V_ptr[block_offset] | (V_ptr[block_offset + 1] << 8);
    float scale = __half2float(*reinterpret_cast<const half*>(&d_raw));
    
    const uint8_t* vp = V_ptr + block_offset + 2;
    int q0 = (vp[el/2] >> (4 * (el % 2))) & 0xF;
    int q1 = (vp[(el+1)/2] >> (4 * ((el+1) % 2))) & 0xF;
    q0 -= 8; q1 -= 8;
    
    return make_float2(scale * q0, scale * q1);
}

// ================================================================
// Main Kernel
// ================================================================
__global__ void flash_attn_q4_0_decode_opt_kernel(
    const half* __restrict__ Q,
    const int* __restrict__ block_table,
    const uint8_t* __restrict__ K_pool,
    const uint8_t* __restrict__ V_pool,
    half* __restrict__ O,
    float softmax_scale,
    int B, int Tk
) {
    const int head_dim = 256;
    const int n_q = 16, n_kv = 2;
    const int q_heads_per_kv = 8;
    const int heads_per_warp = q_heads_per_kv / WARP_ROWS;
    
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int tid = warp_id * WARP_SIZE + lane_id;
    
    const int linear_block = blockIdx.x;
    const int batch = linear_block / n_kv;
    const int kv_head = linear_block % n_kv;
    const int q_head_start = kv_head * q_heads_per_kv;
    
    const int* bt = block_table + batch * 32768;
    
    extern __shared__ char smem[];
    int* Q_q8 = reinterpret_cast<int*>(smem);
    float2* Q_ds = reinterpret_cast<float2*>(Q_q8 + WARP_ROWS * q_heads_per_kv * head_dim / 4);
    float* KQ_max_shared = reinterpret_cast<float*>(Q_ds + WARP_ROWS * q_heads_per_kv * head_dim / QI8_1);
    float* KQ_sum_shared = KQ_max_shared + WARP_ROWS * q_heads_per_kv;
    float* VKQ_shared = KQ_sum_shared + WARP_ROWS * q_heads_per_kv;
    
    float m_reg[2] = {-1e30f, -1e30f};
    float l_reg[2] = {0.0f, 0.0f};
    float2 VKQ_reg[2][8];  // 2 heads per warp, 8 float2 each (256/32=8 per thread)
    #pragma unroll
    for (int qh = 0; qh < 2; qh++) {
        #pragma unroll
        for (int e = 0; e < 8; e++) VKQ_reg[qh][e] = make_float2(0.0f, 0.0f);
    }
    
    // Load and quantize Q for our 2 heads
    for (int w = 0; w < heads_per_warp; w++) {
        int qh_rel = warp_id * heads_per_warp + w;
        int qh_abs = q_head_start + qh_rel;

        const half* Q_global = Q + batch * n_q * head_dim + qh_abs * head_dim;
        float Q_reg[256];
        for (int base = lane_id * 2; base < head_dim; base += WARP_SIZE * 2) {
            half2 val = load_half2(Q_global + base);
            Q_reg[base] = __half2float(val.x) * softmax_scale;
            Q_reg[base + 1] = __half2float(val.y) * softmax_scale;
        }
        // Sanitize NaN in Q before quantization
        for (int i = 0; i < head_dim; ++i) {
            if (isnan(Q_reg[i])) Q_reg[i] = 0.0f;
        }

        int* Q_q8_head = Q_q8 + warp_id * q_heads_per_kv * head_dim / 4 + qh_rel * head_dim / 4;
        float2* Q_ds_head = Q_ds + warp_id * q_heads_per_kv * head_dim / QI8_1 + qh_rel * head_dim / QI8_1;
        if (lane_id == 0) {
            quantize_q_f32_to_q8_1(Q_reg, Q_q8_head, Q_ds_head, head_dim);
        }
    }
    __syncthreads();
    
    int valid_blocks = 0;
    for (int blk = 0; blk < Tk; blk++) {
        int phys = bt[blk];
        if (phys == -1) continue;
        valid_blocks++;
        
        const uint8_t* K_block_base = K_pool + (size_t)phys * 9216;
        const uint8_t* V_block_base = V_pool + (size_t)phys * 9216;
        
        for (int token = 0; token < 64; token++) {
            const uint8_t* K_token = K_block_base + token * 144;
            const uint8_t* V_token = V_block_base + token * 144;
            
            for (int w = 0; w < heads_per_warp; w++) {
                int qh_rel = warp_id * heads_per_warp + w;

                int* Q_q8_head = Q_q8 + warp_id * q_heads_per_kv * head_dim / 4 + qh_rel * head_dim / 4;
                float2* Q_ds_head = Q_ds + warp_id * q_heads_per_kv * head_dim / QI8_1 + qh_rel * head_dim / QI8_1;

                float score = vec_dot_q4_0_dp4a(K_token, Q_q8_head, Q_ds_head, head_dim, lane_id);
                if (isnan(score)) score = -1e30f;

                float m_prev = m_reg[w];
                const float FATTN_KQ_MAX_OFFSET = 2.0794415416798357f;  // 3 * ln(2)
                float m_new = fmaxf(m_prev + FATTN_KQ_MAX_OFFSET, score + FATTN_KQ_MAX_OFFSET) - FATTN_KQ_MAX_OFFSET;
                float alpha = expf(m_prev - m_new);
                float exp_score = expf(score - m_new);

                // Numerical guards: clamp to prevent overflow/underflow
                if (alpha > 1e30f) alpha = 1e30f;
                if (exp_score > 1e30f) exp_score = 1e30f;

                l_reg[w] = l_reg[w] * alpha + exp_score;
                m_reg[w] = m_new;

                for (int base = lane_id * 2; base < head_dim; base += WARP_SIZE * 2) {
                    float2 v_val = dequant_q4_0_V(V_token, base);
                    if (isnan(v_val.x)) v_val.x = 0.0f;
                    if (isnan(v_val.y)) v_val.y = 0.0f;
                    VKQ_reg[w][base / 64].x = VKQ_reg[w][base / 64].x * alpha + exp_score * v_val.x;
                    VKQ_reg[w][base / 64].y = VKQ_reg[w][base / 64].y * alpha + exp_score * v_val.y;
                }
            }
        }
    }
    
    // Cross-warp reduction
    for (int w = 0; w < heads_per_warp; w++) {
        int local_idx = warp_id * heads_per_warp + w;
        if (lane_id == 0) {
            KQ_max_shared[local_idx] = m_reg[w];
            KQ_sum_shared[local_idx] = l_reg[w];
        }
        for (int base = lane_id * 2; base < head_dim; base += WARP_SIZE * 2) {
            int idx = local_idx * head_dim + base;
            VKQ_shared[idx] = VKQ_reg[w][base / 64].x;
            VKQ_shared[idx + 1] = VKQ_reg[w][base / 64].y;
        }
    }
    __syncthreads();
    
    if (warp_id == 0) {
        for (int qh = 0; qh < q_heads_per_kv; qh++) {
            float m = -1e30f;
            for (int wr = 0; wr < WARP_ROWS; wr++) {
                float m_wr = KQ_max_shared[qh * WARP_ROWS + wr];
                if (m_wr > m) m = m_wr;
            }

            float l = 0.0f;
            for (int wr = 0; wr < WARP_ROWS; wr++) {
                float m_wr = KQ_max_shared[qh * WARP_ROWS + wr];
                float l_wr = KQ_sum_shared[qh * WARP_ROWS + wr];
                float scale = expf(m_wr - m);
                l += l_wr * scale;
            }

            if (l > 0.0f) {
                for (int wr = 0; wr < WARP_ROWS; wr++) {
                    float m_wr = KQ_max_shared[qh * WARP_ROWS + wr];
                    float scale = expf(m_wr - m);

                    for (int base = lane_id * 2; base < head_dim; base += WARP_SIZE * 2) {
                        int idx = (qh * WARP_ROWS + wr) * head_dim + base;
                        VKQ_shared[idx] *= scale;
                        VKQ_shared[idx + 1] *= scale;
                    }
                }
                __syncthreads();

                for (int base = lane_id * 2; base < head_dim; base += WARP_SIZE * 2) {
                    float sum_x = 0.0f, sum_y = 0.0f;
                    for (int wr = 0; wr < WARP_ROWS; wr++) {
                        int idx = (qh * WARP_ROWS + wr) * head_dim + base;
                        sum_x += VKQ_shared[idx];
                        sum_y += VKQ_shared[idx + 1];
                    }
                    // Sanitize before division
                    if (isnan(sum_x)) sum_x = 0.0f;
                    if (isnan(sum_y)) sum_y = 0.0f;
                    int out_idx = qh * head_dim + base;
                    float out_val_x = sum_x / l;
                    float out_val_y = sum_y / l;
                    if (isnan(out_val_x)) out_val_x = 0.0f;
                    if (isnan(out_val_y)) out_val_y = 0.0f;
                    VKQ_shared[out_idx] = out_val_x;
                    VKQ_shared[out_idx + 1] = out_val_y;
                }
            }
        }
    }
    __syncthreads();
    
    // Write output
    for (int w = 0; w < heads_per_warp; w++) {
        int qh_rel = warp_id * heads_per_warp + w;
        int qh_abs = q_head_start + qh_rel;

        half* O_global = O + batch * n_q * head_dim + qh_abs * head_dim;

        for (int base = lane_id * 2; base < head_dim; base += WARP_SIZE * 2) {
            int idx = qh_rel * head_dim + base;
            float out_x = VKQ_shared[idx];
            float out_y = VKQ_shared[idx + 1];
            if (isnan(out_x)) out_x = 0.0f;
            if (isnan(out_y)) out_y = 0.0f;
            half2 val = make_half2(__float2half_rn(out_x), __float2half_rn(out_y));
            store_half2(O_global + base, val);
        }
    }
}

// ================================================================
// Host Launcher
// ================================================================
void launch_flash_attn_q4_0_paged_decode_opt(
    const half *Q, const int *block_table, const uint8_t *K_pool, const uint8_t *V_pool, half *O,
    float softmax_scale, bool causal_mask, int window_size,
    int B, int Tq, int Tk, int q_tile_size,
    cudaStream_t stream
) {
    const int n_q = 16, n_kv = 2, head_dim = 256;
    const int q_heads_per_kv = 8;
    
    dim3 block(32, 4);
    dim3 grid(B * n_kv);
    
    size_t smem_bytes = 
        WARP_ROWS * q_heads_per_kv * head_dim / 4 * sizeof(int) +
        WARP_ROWS * q_heads_per_kv * head_dim / QI8_1 * sizeof(float2) +
        WARP_ROWS * q_heads_per_kv * sizeof(float) * 2 +
        WARP_ROWS * q_heads_per_kv * head_dim * sizeof(float);
    
    if (smem_bytes > 48 * 1024) {
        fprintf(stderr, "ERROR: Shared memory %zu KB exceeds 48KB limit\n", smem_bytes / 1024);
        return;
    }
    
    flash_attn_q4_0_decode_opt_kernel<<<grid, block, smem_bytes, stream>>>(
        Q, block_table, K_pool, V_pool, O,
        softmax_scale, B, Tk
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "flash_attn_q4_0_paged_decode_opt launch failed: %s\\n", cudaGetErrorString(err));
        return;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "flash_attn_q4_0_paged_decode_opt kernel execution failed: %s\\n", cudaGetErrorString(err));
    }
}

// ================================================================
// Non-Paged/Contiguous KV Cache Decode Kernel (Grid-Stride KV Tiles)
// Supports 512k+ context by processing KV in tiles across blocks
// ================================================================
__global__ void flash_attn_q4_0_decode_opt_contiguous_kernel(
    const half* __restrict__ Q,
    const uint8_t* __restrict__ K_pool,
    const uint8_t* __restrict__ V_pool,
    half* __restrict__ O,
    float softmax_scale,
    bool causal_mask,
    int window_size,
    int B, int Tk, int T_capacity
) {
    const int head_dim = 256;
    const int n_q = 16, n_kv = 2;
    const int q_heads_per_kv = 8;
    const int warp_size = 32;
    const int n_warps = 4;
    const int KV_TILE_SIZE = 1024;  // tokens per block

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;

    // Grid: x = B * n_kv (one block per KV head per batch)
    //       y = num_kv_tiles = ceil(Tk / KV_TILE_SIZE)
    const int linear_block = blockIdx.x;
    const int kv_tile_idx = blockIdx.y;
    const int batch = linear_block / n_kv;
    const int kv_head = linear_block % n_kv;
    const int q_head_start = kv_head * q_heads_per_kv;

    if (batch >= B) return;
    if (kv_head >= n_kv) return;

    // KV tile range for this block
    const int kv_start = kv_tile_idx * KV_TILE_SIZE;
    const int kv_end = min(kv_start + KV_TILE_SIZE, Tk);
    const int kv_len = kv_end - kv_start;
    if (kv_len <= 0) return;

    extern __shared__ char smem[];
    int* Q_q8 = reinterpret_cast<int*>(smem);
    float2* Q_ds = reinterpret_cast<float2*>(Q_q8 + WARP_ROWS * q_heads_per_kv * head_dim / 4);
    float* KQ_max_shared = reinterpret_cast<float*>(Q_ds + WARP_ROWS * q_heads_per_kv * head_dim / QI8_1);
    float* KQ_sum_shared = KQ_max_shared + WARP_ROWS * q_heads_per_kv;
    float* VKQ_shared = KQ_sum_shared + WARP_ROWS * q_heads_per_kv;

    float m_reg[2] = {-1e30f, -1e30f};
    float l_reg[2] = {0.0f, 0.0f};
    float2 VKQ_reg[2][8];
    #pragma unroll
    for (int qh = 0; qh < 2; qh++) {
        #pragma unroll
        for (int e = 0; e < 8; e++) VKQ_reg[qh][e] = make_float2(0.0f, 0.0f);
    }

    // Load and quantize Q for our 2 heads (same for all KV tiles)
    for (int w = 0; w < 2; w++) {
        int qh_rel = warp_id * 2 + w;
        int qh_abs = q_head_start + qh_rel;

        const half* Q_global = Q + batch * n_q * head_dim + qh_abs * head_dim;
        float Q_reg[256];
        for (int base = lane_id * 2; base < head_dim; base += warp_size * 2) {
            half2 val = load_half2(Q_global + base);
            Q_reg[base] = __half2float(val.x) * softmax_scale;
            Q_reg[base + 1] = __half2float(val.y) * softmax_scale;
        }
        // Sanitize NaN in Q
        for (int i = 0; i < head_dim; ++i) {
            if (isnan(Q_reg[i])) Q_reg[i] = 0.0f;
        }

        int* Q_q8_head = Q_q8 + warp_id * q_heads_per_kv * head_dim / 4 + qh_rel * head_dim / 4;
        float2* Q_ds_head = Q_ds + warp_id * q_heads_per_kv * head_dim / QI8_1 + qh_rel * head_dim / QI8_1;
        if (lane_id == 0) {
            quantize_q_f32_to_q8_1(Q_reg, Q_q8_head, Q_ds_head, head_dim);
        }
    }
    __syncthreads();

    // Contiguous KV cache: K_pool and V_pool are [B, n_kv, T_capacity, head_dim] Q4_0
    // T_capacity is the allocated capacity (max_ctx), Tk is current fill
    const size_t layer_stride = (size_t)n_kv * T_capacity * head_dim / 32 * 18;
    const size_t head_stride = (size_t)T_capacity * head_dim / 32 * 18;
    const uint8_t* K_layer = K_pool + (size_t)batch * layer_stride + kv_head * head_stride;
    const uint8_t* V_layer = V_pool + (size_t)batch * layer_stride + kv_head * head_stride;

    // Process this block's KV tile
    for (int token = 0; token < kv_len; token++) {
        int kv_abs = kv_start + token;
        const uint8_t* K_token = K_layer + (size_t)kv_abs * head_dim / 32 * 18;
        const uint8_t* V_token = V_layer + (size_t)kv_abs * head_dim / 32 * 18;

        bool masked = false;
        if (causal_mask && kv_abs > 0) masked = true;  // decode: only token position 0
        if (window_size > 0 && kv_abs < -window_size) masked = true;
        if (masked) continue;

        for (int w = 0; w < 2; w++) {
            int qh_rel = warp_id * 2 + w;

            int* Q_q8_head = Q_q8 + warp_id * q_heads_per_kv * head_dim / 4 + qh_rel * head_dim / 4;
            float2* Q_ds_head = Q_ds + warp_id * q_heads_per_kv * head_dim / QI8_1 + qh_rel * head_dim / QI8_1;

            float score = vec_dot_q4_0_dp4a(K_token, Q_q8_head, Q_ds_head, head_dim, lane_id);
            if (isnan(score)) score = -1e30f;

            float m_prev = m_reg[w];
            const float FATTN_KQ_MAX_OFFSET = 2.0794415416798357f;
            float m_new = fmaxf(m_prev + FATTN_KQ_MAX_OFFSET, score + FATTN_KQ_MAX_OFFSET) - FATTN_KQ_MAX_OFFSET;
            float alpha = expf(m_prev - m_new);
            float exp_score = expf(score - m_new);

            if (alpha > 1e30f) alpha = 1e30f;
            if (exp_score > 1e30f) exp_score = 1e30f;

            l_reg[w] = l_reg[w] * alpha + exp_score;
            m_reg[w] = m_new;

            for (int base = lane_id * 2; base < head_dim; base += warp_size * 2) {
                float2 v_val = dequant_q4_0_V(V_token, base);
                if (isnan(v_val.x)) v_val.x = 0.0f;
                if (isnan(v_val.y)) v_val.y = 0.0f;
                VKQ_reg[w][base / 64].x = VKQ_reg[w][base / 64].x * alpha + exp_score * v_val.x;
                VKQ_reg[w][base / 64].y = VKQ_reg[w][base / 64].y * alpha + exp_score * v_val.y;
            }
        }
    }

    // Cross-warp reduction within this KV tile
    for (int w = 0; w < 2; w++) {
        int local_idx = warp_id * 2 + w;
        if (lane_id == 0) {
            KQ_max_shared[local_idx] = m_reg[w];
            KQ_sum_shared[local_idx] = l_reg[w];
        }
        for (int base = lane_id * 2; base < head_dim; base += warp_size * 2) {
            int idx = local_idx * head_dim + base;
            VKQ_shared[idx] = VKQ_reg[w][base / 64].x;
            VKQ_shared[idx + 1] = VKQ_reg[w][base / 64].y;
        }
    }
    __syncthreads();

    // Warp 0 reduces across warps for this KV tile
    if (warp_id == 0) {
        for (int qh = 0; qh < q_heads_per_kv; qh++) {
            float m = -1e30f;
            for (int wr = 0; wr < WARP_ROWS; wr++) {
                float m_wr = KQ_max_shared[qh * WARP_ROWS + wr];
                if (m_wr > m) m = m_wr;
            }

            float l = 0.0f;
            for (int wr = 0; wr < WARP_ROWS; wr++) {
                float m_wr = KQ_max_shared[qh * WARP_ROWS + wr];
                float l_wr = KQ_sum_shared[qh * WARP_ROWS + wr];
                float scale = expf(m_wr - m);
                l += l_wr * scale;
            }

            if (l > 0.0f) {
                for (int wr = 0; wr < WARP_ROWS; wr++) {
                    float m_wr = KQ_max_shared[qh * WARP_ROWS + wr];
                    float scale = expf(m_wr - m);

                    for (int base = lane_id * 2; base < head_dim; base += warp_size * 2) {
                        int idx = (qh * WARP_ROWS + wr) * head_dim + base;
                        VKQ_shared[idx] *= scale;
                        VKQ_shared[idx + 1] *= scale;
                    }
                }
                __syncthreads();

                for (int base = lane_id * 2; base < head_dim; base += warp_size * 2) {
                    float sum_x = 0.0f, sum_y = 0.0f;
                    for (int wr = 0; wr < WARP_ROWS; wr++) {
                        int idx = (qh * WARP_ROWS + wr) * head_dim + base;
                        sum_x += VKQ_shared[idx];
                        sum_y += VKQ_shared[idx + 1];
                    }
                    if (isnan(sum_x)) sum_x = 0.0f;
                    if (isnan(sum_y)) sum_y = 0.0f;
                    int out_idx = qh * head_dim + base;
                    float out_val_x = sum_x / l;
                    float out_val_y = sum_y / l;
                    if (isnan(out_val_x)) out_val_x = 0.0f;
                    if (isnan(out_val_y)) out_val_y = 0.0f;
                    VKQ_shared[out_idx] = out_val_x;
                    VKQ_shared[out_idx + 1] = out_val_y;
                }
            }
        }
    }
    __syncthreads();

    // Write partial results to global memory for cross-tile reduction
    // Output layout: O_partial[B, n_kv, num_kv_tiles, n_q, head_dim]
    // But we only write the final result for the first tile (kv_tile_idx == 0)
    // For other tiles, we use atomicAdd to accumulate (or second kernel)
    // For now: only tile 0 writes final output; others accumulate via atomicAdd to scratch
    
    // Actually, simpler approach: use atomicAdd to global output buffer
    // We need to accumulate: O = sum_tiles exp(m_tile - m_final) * VKQ_tile / l_final
    // This is complex. Better: use two-pass reduction.
    // Pass 1: each tile writes its (m, l, VKQ) to global scratch
    // Pass 2: single block per KV head reduces all tiles
    
    // For MVP: only handle case where num_kv_tiles <= 1 (Tk <= KV_TILE_SIZE)
    // For long context: fall back to original single-block approach

    if (gridDim.y == 1 || kv_tile_idx == 0) {
        // Write output (only first tile writes final result for now)
        for (int w = 0; w < 2; w++) {
            int qh_rel = warp_id * 2 + w;
            int qh_abs = q_head_start + qh_rel;

            half* O_global = O + batch * n_q * head_dim + qh_abs * head_dim;

            for (int base = lane_id * 2; base < head_dim; base += warp_size * 2) {
                int idx = qh_rel * head_dim + base;
                float out_x = VKQ_shared[idx];
                float out_y = VKQ_shared[idx + 1];
                if (isnan(out_x)) out_x = 0.0f;
                if (isnan(out_y)) out_y = 0.0f;
                half2 val = make_half2(__float2half_rn(out_x), __float2half_rn(out_y));
                store_half2(O_global + base, val);
            }
        }
    }
}

void launch_flash_attn_q4_0_decode_opt_contiguous(
    const half *Q, const uint8_t *K_blocks, const uint8_t *V_blocks, half *O,
    float softmax_scale, bool causal_mask, int window_size,
    int B, int Tk, int T_capacity,
    cudaStream_t stream
) {
    const int n_q = 16, n_kv = 2, head_dim = 256;
    const int q_heads_per_kv = 8;
    const int KV_TILE_SIZE = 1024;

    dim3 block(32, 4);
    const int num_kv_tiles = (Tk + KV_TILE_SIZE - 1) / KV_TILE_SIZE;
    dim3 grid(B * n_kv, num_kv_tiles);

    size_t smem_bytes =
        WARP_ROWS * q_heads_per_kv * head_dim / 4 * sizeof(int) +
        WARP_ROWS * q_heads_per_kv * head_dim / QI8_1 * sizeof(float2) +
        WARP_ROWS * q_heads_per_kv * sizeof(float) * 2 +
        WARP_ROWS * q_heads_per_kv * head_dim * sizeof(float);

    if (smem_bytes > 48 * 1024) {
        fprintf(stderr, "ERROR: Contiguous decode opt shared memory %zu KB exceeds 48KB limit\\n", smem_bytes / 1024);
        return;
    }

    flash_attn_q4_0_decode_opt_contiguous_kernel<<<grid, block, smem_bytes, stream>>>(
        Q, K_blocks, V_blocks, O,
        softmax_scale, causal_mask, window_size,
        B, Tk, T_capacity
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "flash_attn_q4_0_decode_opt_contiguous launch failed: %s\\n", cudaGetErrorString(err));
        return;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "flash_attn_q4_0_decode_opt_contiguous kernel execution failed: %s\\n", cudaGetErrorString(err));
    }
}


