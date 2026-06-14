/**
 * rotorquant.cu — RotorQuant Pre-Quantization Rotation Kernels
 * 
 * Implements block-diagonal Clifford rotors for TurboQuant+ style compression:
 * - 2×2 Givens rotations (PlanarQuant) for K cache
 * - 4×4 Isoclinic quaternion rotations for V cache
 * 
 * Based on Scrya Research "RotorQuant" (PlanarQuant + IsoQuant)
 * Benefits: 44× fewer params than WHT, 28% faster decode
 */

#include "wubu_turboquant.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>

// ================================================================
// Host: Pre-compute Givens/Quaternion tables per layer
// ================================================================

__global__ void init_givens_table_kernel(
    const float* optimal_angles,  // [n_layers * 128] per-head optimal angles
    rotor_tables_t* tables,
    int n_layers
) {
    int layer = blockIdx.x;
    int pair = threadIdx.x;
    int n_pairs = 128;  // 256 dim / 2
    
    if (layer >= n_layers || pair >= n_pairs) return;
    
    // For each KV head
    for (int h = 0; h < 2; h++) {
        float angle = optimal_angles[layer * n_pairs + pair];
        tables->cos[h][pair] = cosf(angle);
        tables->sin[h][pair] = sinf(angle);
        
        // Quaternion for V: isoclinic rotation (4 params per 4 elements)
        // From angle, construct isoclinic rotation
        // Simple: use same angle for all 4 quaternion params
        tables->quat[h][pair/2][pair%2] = angle;
    }
}

void rotorquant_init(rotor_tables_t* h_tables, rotor_tables_t* d_tables,
                     int n_layers, const float* optimal_angles) {
    // Copy to device
    cudaMemcpy(d_tables, h_tables, sizeof(rotor_tables_t), cudaMemcpyHostToDevice);
    // Or init on device
    init_givens_table_kernel<<<n_layers, 128>>>(optimal_angles, d_tables, n_layers);
    cudaDeviceSynchronize();
}

// ================================================================
// Kernel: Givens Rotation + Q4_0 Quantize (K cache)
// 16 pairs per warp (256 threads = 8 warps, 4 blocks = 8 warps × 2 pairs)
// ================================================================

__global__ void givens_rotate_q40_kernel(
    const float* __restrict__ input,   // [n_layers * 2 * 256] float
    block_q4_0_t* __restrict__ output, // [n_layers * 2 * 8] block_q4_0_t
    const rotor_tables_t* tables,
    int n_layers
) {
    int layer = blockIdx.x;
    int h = blockIdx.y;  // 0=KV head 0, 1=KV head 1
    int pair = threadIdx.x;  // 0..127 (16 pairs per 32 elements, 8 blocks)
    
    if (layer >= n_layers || pair >= 128) return;
    
    // 16 pairs per 32 elements, 8 blocks = 128 pairs total per head
    const float* in = input + layer * 2 * 256 + h * 256 + pair * 2;
    float a = in[0];
    float b = in[1];
    
    float c = tables->cos[h][pair];
    float s = tables->sin[h][pair];
    
    // Rotate: [a', b'] = [c*a - s*b, s*a + c*b]
    float a_rot = c * a - s * b;
    float b_rot = s * a + c * b;
    
    // Quantize pair to Q4_0 (in shared memory, then write)
    __shared__ float sh_input[256];
    __shared__ block_q4_0_t sh_output[8];
    
    sh_input[threadIdx.x * 2] = a_rot;
    sh_input[threadIdx.x * 2 + 1] = b_rot;
    __syncthreads();
    
    // First thread in each 32-element block quantizes
    if ((threadIdx.x & 15) == 0) {
        int base = (threadIdx.x / 16) * 32;
        float block_data[32];
        for (int i = 0; i < 32; i++) block_data[i] = sh_input[base + i];
        
        block_q4_0_t qblock;
        // Q4_0 quantize (inline)
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            float ax = fabsf(block_data[i]);
            if (ax > amax) amax = ax;
        }
        if (amax > 0.0f) {
            float d = amax / 7.0f;
            qblock.d = __float2half_rn(d);
            float id = 1.0f / d;
            for (int i = 0; i < 32; i++) {
                int q = (int)(block_data[i] * id + 8.0f);
                if (q < 0) q = 0;
                if (q > 15) q = 15;
                qblock.qs[i / 2] |= (uint8_t)(q << (4 * (i % 2)));
            }
        } else {
            qblock.d = 0;
            memset(qblock.qs, 0, 16);
        }
        sh_output[threadIdx.x / 16] = qblock;
    }
    __syncthreads();
    
    // Write to global
    int out_idx = layer * 2 * 8 + h * 8 + threadIdx.x / 16;
    output[out_idx] = sh_output[threadIdx.x / 16];
}

// ================================================================
// Kernel: Quaternion Rotation + Q2_0 Quantize (V cache)
// 4×4 Isoclinic rotation per 4-element group
// ================================================================

__device__ __forceinline__ void quat_apply_4(const float* x, const float* q, float* out) {
    // Isoclinic quaternion rotation: q = [a, b, c, d]
    // Rotate 4 elements as pair of complex planes
    // x = [w, x, y, z] as quaternion
    // out = q * x * q^(-1) (isoclinic: left and right same)
    
    // Simplified: treat as 2×2 complex rotations
    float a = q[0], b = q[1], c = q[2], d = q[3];
    float w = x[0], x1 = x[1], y = x[2], z = x[3];
    
    // q * x quaternion multiplication
    float w2 = a*w - b*x1 - c*y - d*z;
    float x2 = a*x1 + b*w + c*z - d*y;
    float y2 = a*y - b*z + c*w + d*x1;
    float z2 = a*z + b*y - c*x1 + d*w;
    
    // x * q^(-1) (conjugate since unit)
    out[0] =  w2*a + x2*b + y2*c + z2*d;
    out[1] = -w2*b + x2*a - y2*d + z2*c;
    out[2] =  w2*c + x2*d - y2*a + z2*b;
    out[3] = -w2*d + x2*c + y2*b - z2*a;
}

__global__ void quaternion_rotate_q20_kernel(
    const float* __restrict__ input,   // [n_layers * 2 * 256] float
    block_q2_0_t* __restrict__ output, // [n_layers * 2 * 8] block_q2_0_t
    const rotor_tables_t* tables,
    int n_layers
) {
    int layer = blockIdx.x;
    int h = blockIdx.y;
    int group = threadIdx.x;  // 0..63 (4-element groups)
    
    if (layer >= n_layers || group >= 64) return;
    
    const float* in = input + layer * 2 * 256 + h * 256 + group * 4;
    const float* q = tables->quat[h][group];
    
    float rotated[4];
    quat_apply_4(in, q, rotated);
    
    __shared__ float sh_rotated[256];
    sh_rotated[threadIdx.x * 4] = rotated[0];
    sh_rotated[threadIdx.x * 4 + 1] = rotated[1];
    sh_rotated[threadIdx.x * 4 + 2] = rotated[2];
    sh_rotated[threadIdx.x * 4 + 3] = rotated[3];
    __syncthreads();
    
    // First thread in each 32-element block quantizes (8 threads per block)
    if ((threadIdx.x & 7) == 0) {
        int base = (threadIdx.x / 8) * 32;
        float block_data[32];
        for (int i = 0; i < 32; i++) block_data[i] = sh_rotated[base + i];
        
        block_q2_0_t qblock;
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            float ax = fabsf(block_data[i]);
            if (ax > amax) amax = ax;
        }
        if (amax > 0.0f) {
            float d = amax / 1.0f;
            qblock.d = __float2half_rn(d);
            float id = 1.0f / d;
            for (int i = 0; i < 32; i++) {
                int q_val = (int)(block_data[i] * id + 2.0f);
                if (q_val < 0) q_val = 0;
                if (q_val > 3) q_val = 3;
                qblock.qs[i / 4] |= (uint8_t)(q_val << (2 * (i % 4)));
            }
        } else {
            qblock.d = 0;
            memset(qblock.qs, 0, 8);
        }
        
        // Quantize 32 elements = 4 Q2_0 blocks
        output[(blockIdx.x * 2 + h) * 8 * 4 + threadIdx.x / 8 * 4 + threadIdx.x % 8] = qblock;
    }
}

// ================================================================
// Host launchers
// ================================================================

void rotorquant_givens_q40(const float* d_input, block_q4_0_t* d_output,
                           const rotor_tables_t* d_tables, int n_layers) {
    dim3 grid(n_layers, 2);  // 2 KV heads
    dim3 block(128);  // 128 pairs per head
    givens_rotate_q40_kernel<<<grid, block>>>(d_input, d_output, d_tables, n_layers);
    cudaDeviceSynchronize();
}

void rotorquant_quat_q20(const float* d_input, block_q2_0_t* d_output,
                         const rotor_tables_t* d_tables, int n_layers) {
    dim3 grid(n_layers, 2);
    dim3 block(64);  // 64 groups of 4
    quaternion_rotate_q20_kernel<<<grid, block>>>(d_input, d_output, d_tables, n_layers);
    cudaDeviceSynchronize();
}

// ================================================================
// TurboQuant: In-kernel WHT + Quantize fusion
// ================================================================

__global__ void turboquant_wht_q40_kernel(
    const float* __restrict__ input,
    block_q4_0_t* __restrict__ output,
    int n_blocks  // total 32-element blocks
) {
    int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_id >= n_blocks) return;
    
    const float* in = input + block_id * 32;
    block_q4_0_t* out = output + block_id;
    
    // Load into shared memory
    __shared__ float sh_data[32];
    if (threadIdx.x < 32) {
        sh_data[threadIdx.x] = in[threadIdx.x];
    }
    __syncthreads();
    
    // In-place WHT (butterfly)
    for (int stride = 1; stride < 32; stride <<= 1) {
        if (threadIdx.x < 32) {
            int pair = threadIdx.x ^ stride;
            if (threadIdx.x < pair) {
                float u = sh_data[threadIdx.x];
                float v = sh_data[pair];
                sh_data[threadIdx.x] = u + v;
                sh_data[pair] = u - v;
            }
        }
        __syncthreads();
    }
    
    // First thread quantizes block
    if (threadIdx.x == 0) {
        block_q4_0_t qblock;
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            float ax = fabsf(sh_data[i]);
            if (ax > amax) amax = ax;
        }
        if (amax > 0.0f) {
            float d = amax / 7.0f;
            qblock.d = __float2half_rn(d);
            float id = 1.0f / d;
            for (int i = 0; i < 32; i++) {
                int q = (int)(sh_data[i] * id + 8.0f);
                if (q < 0) q = 0;
                if (q > 15) q = 15;
                qblock.qs[i / 2] |= (uint8_t)(q << (4 * (i % 2)));
            }
        } else {
            qblock.d = 0;
            memset(qblock.qs, 0, 16);
        }
        *out = qblock;
    }
}

__global__ void turboquant_wht_q20_kernel(
    const float* __restrict__ input,
    block_q2_0_t* __restrict__ output,
    int n_blocks
) {
    int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_id >= n_blocks) return;
    
    const float* in = input + block_id * 32;
    block_q2_0_t* out = output + block_id;
    
    __shared__ float sh_data[32];
    if (threadIdx.x < 32) {
        sh_data[threadIdx.x] = in[threadIdx.x];
    }
    __syncthreads();
    
    // WHT
    for (int stride = 1; stride < 32; stride <<= 1) {
        if (threadIdx.x < 32) {
            int pair = threadIdx.x ^ stride;
            if (threadIdx.x < pair) {
                float u = sh_data[threadIdx.x];
                float v = sh_data[pair];
                sh_data[threadIdx.x] = u + v;
                sh_data[pair] = u - v;
            }
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        block_q2_0_t qblock;
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            float ax = fabsf(sh_data[i]);
            if (ax > amax) amax = ax;
        }
        if (amax > 0.0f) {
            float d = amax / 1.0f;
            qblock.d = __float2half_rn(d);
            float id = 1.0f / d;
            for (int i = 0; i < 32; i++) {
                int q = (int)(sh_data[i] * id + 2.0f);
                if (q < 0) q = 0;
                if (q > 3) q = 3;
                qblock.qs[i / 4] |= (uint8_t)(q << (2 * (i % 4)));
            }
        } else {
            qblock.d = 0;
            memset(qblock.qs, 0, 8);
        }
        *out = qblock;
    }
}

void turboquant_wht_q40(const float* d_input, block_q4_0_t* d_output, int n_blocks) {
    dim3 grid((n_blocks + 255) / 256);
    dim3 block(256);
    turboquant_wht_q40_kernel<<<grid, block>>>(d_input, d_output, n_blocks);
    cudaDeviceSynchronize();
}

void turboquant_wht_q20(const float* d_input, block_q2_0_t* d_output, int n_blocks) {
    dim3 grid((n_blocks + 255) / 256);
    dim3 block(256);
    turboquant_wht_q20_kernel<<<grid, block>>>(d_input, d_output, n_blocks);
    cudaDeviceSynchronize();
}