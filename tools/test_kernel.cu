// Minimal kernel test - tests flash_attn_q4_0_decode_opt kernel without full model loading
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include <cstdio>
#include <cmath>
#include <cstdint>

#define WARP_SIZE 32
#define WARP_ROWS 4
#define QK4_0 32
#define QI8_1 32

typedef struct { uint16_t d; uint8_t qs[QK4_0/2]; } block_q4_0;

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

__device__ void quantize_q_f32_to_q8_1(const float* src, int* dst_i32, float2* dst_ds, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = fabsf(src[i]);
        if (v > max_abs) max_abs = v;
    }
    float q_scale = max_abs / 127.0f;
    if (q_scale == 0.0f) q_scale = 1.0f;
    
    for (int i = 0; i < n / QI8_1; i++) {
        int32_t sum = 0;
        int32_t isum = 0;
        for (int j = 0; j < QI8_1; j++) {
            int8_t q = __float2int_rn(src[i * QI8_1 + j] / q_scale);
            sum |= (q & 0xFF) << (8 * j);
            isum += q;
        }
        dst_i32[i] = sum;
        dst_ds[i] = make_float2(q_scale, isum * q_scale / QI8_1);
    }
}

__device__ __forceinline__ float vec_dot_q4_0_dp4a(const uint8_t* K_block, const int* Q_q8, const float2* Q_ds, int head_dim, int lane_id) {
    // Use byte-wise access to avoid alignment issues
    const uint8_t* K_ptr = K_block;
    float sum = 0.0f;
    
    for (int k_KQ_0 = lane_id; k_KQ_0 < head_dim / 4; k_KQ_0 += WARP_SIZE) {
        const int ib    = (k_KQ_0 * 4) / QK4_0;
        const int iqs4  = (k_KQ_0 * 4) % (QK4_0/2);
        const int shift = ((k_KQ_0 * 4) % QK4_0) / (QK4_0/2);
        
        // Manually read block: 2 bytes scale + 16 bytes qs
        size_t block_offset = ib * 18;
        // Read scale (2 bytes, little-endian)
        uint16_t d_raw = K_ptr[block_offset] | (K_ptr[block_offset + 1] << 8);
        float k_scale = __half2float(*reinterpret_cast<const half*>(&d_raw));
        
        // Read 4 qs bytes (avoid int cast for alignment)
        const uint8_t* qs_ptr = K_ptr + block_offset + 2 + 4 * iqs4;
        int v = qs_ptr[0] | (qs_ptr[1] << 8) | (qs_ptr[2] << 16) | (qs_ptr[3] << 24);
        v = (v >> (4 * shift)) & 0x0F0F0F0F;
        
        int k_int8 = v - 0x08080808;
        
        int u = Q_q8[k_KQ_0];
        int sumi = __dp4a(u, k_int8, 0);
        
        const float2 Q_ds_val = Q_ds[k_KQ_0 / QI8_1];
        
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
    float scale = __half2float(*reinterpret_cast<const half*>(&d_raw));
    
    const uint8_t* vp = V_ptr + block_offset + 2;
    int q0 = (vp[el/2] >> (4 * (el % 2))) & 0xF;
    int q1 = (vp[(el+1)/2] >> (4 * ((el+1) % 2))) & 0xF;
    q0 -= 8; q1 -= 8;
    
    return make_float2(scale * q0, scale * q1);
}

__global__ void test_kernel(
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
    const int heads_per_warp = 2;
    
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
    float2 VKQ_reg[2][4];
    #pragma unroll
    for (int qh = 0; qh < 2; qh++) {
        #pragma unroll
        for (int e = 0; e < 4; e++) VKQ_reg[qh][e] = make_float2(0.0f, 0.0f);
    }
    
    // Load and quantize Q
    for (int w = 0; w < heads_per_warp; w++) {
        int qh_rel = warp_id * heads_per_warp + w;
        int qh_abs = q_head_start + qh_rel;
        
        const half* Q_global = Q + batch * n_q * head_dim + qh_abs * head_dim;
        float Q_reg[256];
        // Use even indices only for half2 loads (4-byte aligned)
        for (int base = lane_id * 2; base < head_dim; base += WARP_SIZE * 2) {
            half2 val = load_half2(Q_global + base);
            Q_reg[base] = __half2float(val.x) * softmax_scale;
            Q_reg[base + 1] = __half2float(val.y) * softmax_scale;
        }
        
        int* Q_q8_head = Q_q8 + warp_id * q_heads_per_kv * head_dim / 4 + qh_rel * head_dim / 4;
        float2* Q_ds_head = Q_ds + warp_id * q_heads_per_kv * head_dim / QI8_1 + qh_rel * head_dim / QI8_1;
        if (lane_id == 0) {
            quantize_q_f32_to_q8_1(Q_reg, Q_q8_head, Q_ds_head, head_dim);
        }
    }
    __syncthreads();
    
    // Main loop
    if (tid == 0) printf("Kernel start: batch=%d kv_head=%d Tk=%d\n", batch, kv_head, Tk);
    
    int valid_blocks = 0;
    for (int blk = 0; blk < Tk; blk++) {
        int phys = bt[blk];
        if (phys == -1) continue;
        
        if (tid == 0 && valid_blocks == 0) printf("First valid block: blk=%d phys=%d\n", blk, phys);
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
                
                float m_prev = m_reg[w];
                float m_new = fmaxf(m_prev + 4.0f, score + 4.0f) - 4.0f;
                float alpha = expf(m_prev - m_new);
                float exp_score = expf(score - m_new);
                
                l_reg[w] = l_reg[w] * alpha + exp_score;
                m_reg[w] = m_new;
                
                // Accumulate V in registers - use aligned accesses
                for (int base = lane_id * 2; base < head_dim; base += WARP_SIZE * 2) {
                    float2 v_val = dequant_q4_0_V(V_token, base);
                    VKQ_reg[w][base / 64].x = VKQ_reg[w][base / 64].x * alpha + exp_score * v_val.x;
                    VKQ_reg[w][base / 64].y = VKQ_reg[w][base / 64].y * alpha + exp_score * v_val.y;
                }
            }
        }
    }
    
    if (tid == 0) printf("Processed %d valid blocks\n", valid_blocks);
    
    // Cross-warp reduction
    for (int w = 0; w < heads_per_warp; w++) {
        int local_idx = warp_id * heads_per_warp + w;
        if (lane_id == 0) {
            KQ_max_shared[local_idx] = m_reg[w];
            KQ_sum_shared[local_idx] = l_reg[w];
        }
        // Store VKQ to shared memory - aligned access
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
                    int out_idx = qh * head_dim + base;
                    VKQ_shared[out_idx] = sum_x / l;
                    VKQ_shared[out_idx + 1] = sum_y / l;
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
        
        // Use even indices for half2 stores
        for (int base = lane_id * 2; base < head_dim; base += WARP_SIZE * 2) {
            int idx = qh_rel * head_dim + base;
            half2 val = make_half2(__float2half_rn(VKQ_shared[idx]), __float2half_rn(VKQ_shared[idx + 1]));
            store_half2(O_global + base, val);
        }
    }
}

int main() {
    // Test parameters
    const int B = 1, head_dim = 256, n_q = 16, n_kv = 2;
    const int Tk = 4;  // 4 KV blocks = 256 tokens
    
    printf("Testing flash_attn_q4_0_decode_opt kernel...\n");
    
    // Allocate device memory
    half *d_Q, *d_O;
    int *d_block_table;
    uint8_t *d_K_pool, *d_V_pool;
    
    cudaMalloc(&d_Q, B * n_q * head_dim * sizeof(half));
    cudaMalloc(&d_O, B * n_q * head_dim * sizeof(half));
    cudaMalloc(&d_block_table, B * 32768 * sizeof(int));
    cudaMalloc(&d_K_pool, 32768 * 9216);  // 32768 blocks * 9216 bytes
    cudaMalloc(&d_V_pool, 32768 * 9216);
    
    // Initialize Q with random values
    half* h_Q = new half[B * n_q * head_dim];
    for (int i = 0; i < B * n_q * head_dim; i++) {
        h_Q[i] = __float2half_rn((float)rand() / RAND_MAX * 2 - 1);
    }
    cudaMemcpy(d_Q, h_Q, B * n_q * head_dim * sizeof(half), cudaMemcpyHostToDevice);
    
    // Initialize K_pool with random Q4_0 data
    uint8_t* h_K = new uint8_t[32768 * 9216];
    for (int i = 0; i < 32768 * 9216; i++) {
        h_K[i] = rand() % 256;
    }
    cudaMemcpy(d_K_pool, h_K, 32768 * 9216, cudaMemcpyHostToDevice);
    
    // Initialize V_pool with random Q4_0 data
    uint8_t* h_V = new uint8_t[32768 * 9216];
    for (int i = 0; i < 32768 * 9216; i++) {
        h_V[i] = rand() % 256;
    }
    cudaMemcpy(d_V_pool, h_V, 32768 * 9216, cudaMemcpyHostToDevice);
    
    // Block table: first Tk blocks valid, rest -1
    int* h_block_table = new int[B * 32768];
    for (int i = 0; i < B * 32768; i++) {
        h_block_table[i] = (i < Tk) ? i : -1;
    }
    cudaMemcpy(d_block_table, h_block_table, B * 32768 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 block(32, 4);  // 128 threads
    dim3 grid(B * n_kv);  // B * n_kv = 2 blocks
    
    size_t smem_bytes = 
        WARP_ROWS * 8 * head_dim / 4 * sizeof(int) +
        WARP_ROWS * 8 * head_dim / QI8_1 * sizeof(float2) +
        WARP_ROWS * 8 * sizeof(float) * 2 +
        WARP_ROWS * 8 * head_dim * sizeof(float);
    
    printf("Shared memory: %zu KB\n", smem_bytes / 1024);
    
    test_kernel<<<grid, block, smem_bytes, 0>>>(
        d_Q, d_block_table, d_K_pool, d_V_pool, d_O,
        0.0625f, B, Tk
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("Kernel completed successfully!\n");
    
    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_O);
    cudaFree(d_block_table);
    cudaFree(d_K_pool);
    cudaFree(d_V_pool);
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_block_table;
    
    return 0;
}
