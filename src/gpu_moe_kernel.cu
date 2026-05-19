/**
 * gpu_moe_kernel.cu — GPU MoE expert compute for IQ2_XXS/IQ3_XXS quantized weights.
 *
 * Uses __constant__ memory for IQ2_XXS grid lookup table (2KB) and
 * ksigns_iq2xs sign table (128 bytes).
 *
 * Each thread block handles one expert with D_FF=512 threads:
 *   Phase 1: Each thread computes gate[idx] and up[idx], writes act[idx]=SiLU(gate)*up to shmem
 *   Phase 2: Each thread computes 4 down matmul elements using shared act[] array
 *   Phase 3: Each thread adds weighted result to output
 *
 * Shared memory layout: act[D_FF] + x[D_MODEL] = 512+2048 = 2560 floats = 10KB
 */
#include "gpu_moe_kernel.h"
#include "gguf_reader.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define QK_K 256
#define D_FF 512
#define D_MODEL 2048

// ================================================================
// Constant memory: IQ2_XXS lookup tables
// ================================================================
static __constant__ uint64_t d_iq2xxs_grid[256];
static __constant__ uint8_t  d_ksigns_iq2xs[128];
static const uint8_t h_kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};
static __constant__ uint8_t d_kmask_iq2xs[8];

// Host-side tables
#include "iq2xxs_grid_data.inc"

// ================================================================
// Device helpers
// ================================================================
static __device__ __inline__ float d_f16_f32(uint16_t v) {
    int s=(v>>15)&1,e=(v>>10)&0x1F,m=v&0x3FF; uint32_t f;
    if(e==0)f=(s<<31);else if(e==31)f=(s<<31)|(0xFF<<23)|(m<<13);
    else f=(s<<31)|((127-15+e)<<23)|(m<<13);
    float r; __builtin_memcpy(&r,&f,4); return r;
}

// Fused dot product: one IQ2_XXS block (66 bytes → 256 values) with x[0..255]
// No intermediate F32 array — dequant and dot in one loop.
static __device__ float iq2_xxs_dot_block_gpu(const uint8_t *block, const float *x) {
    float d = d_f16_f32(*(const uint16_t *)block);
    const uint16_t *qs16 = (const uint16_t *)(block + 2);
    float total = 0.0f;
    for (int ib32 = 0; ib32 < QK_K/32; ib32++) {
        uint32_t aux32[2];
        // Load 8 bytes from qs16 + 4*ib32 (may not be 4-byte aligned — use byte copy)
        const uint8_t *src = (const uint8_t*)(qs16 + 4*ib32);
        aux32[0] = ((uint32_t)src[0]) | ((uint32_t)src[1]<<8) | ((uint32_t)src[2]<<16) | ((uint32_t)src[3]<<24);
        aux32[1] = ((uint32_t)src[4]) | ((uint32_t)src[5]<<8) | ((uint32_t)src[6]<<16) | ((uint32_t)src[7]<<24);
        const uint8_t *aux8 = (const uint8_t *)aux32;
        float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            const uint8_t *grid = (const uint8_t*)(&d_iq2xxs_grid[aux8[l]]);
            uint8_t signs = d_ksigns_iq2xs[(aux32[1] >> (7*l)) & 127];
            int base = ib32 * 32 + l * 8;
            for (int j = 0; j < 8; j++) {
                float val = db * (float)grid[j];
                if (signs & d_kmask_iq2xs[j])
                    total -= x[base + j] * val;
                else
                    total += x[base + j] * val;
            }
        }
    }
    return total;
}

// ================================================================
// MoE expert kernel: ONE block handles ONE expert
// gridDim = 1 (one expert), blockDim = D_FF (512)
// Shared memory: act[D_FF] + x[D_MODEL] (extern __shared__)
// ================================================================
__global__ void moe_single_expert_kernel(
    const float * __restrict__ x,        // [D_MODEL] input
    const uint8_t * __restrict__ gate_q, // [gate_bytes_per_expert]
    const uint8_t * __restrict__ up_q,   // [up_bytes_per_expert]
    const uint8_t * __restrict__ down_q, // [down_bytes_per_expert]
    float weight,                        // routing weight for this expert
    float * __restrict__ output          // [D_MODEL] output (write, no atomics needed)
) {
    extern __shared__ float smem[];
    float *s_act = smem;
    float *s_x   = smem + D_FF;

    int idx = threadIdx.x;

    // Load x
    for (int i = threadIdx.x; i < D_MODEL; i += blockDim.x)
        s_x[i] = x[i];
    __syncthreads();

    // Phase 1: gate matmul
    const int D_BPC = (D_MODEL + QK_K - 1) / QK_K;
    const int BLK_SZ = 66;
    double g_sum = 0.0;
    const uint8_t *g_col = gate_q + (int64_t)idx * D_BPC * BLK_SZ;
    for (int b = 0; b < D_BPC; b++)
        g_sum += (double)iq2_xxs_dot_block_gpu(g_col + b * BLK_SZ, s_x + b * QK_K);

    // Phase 1b: up matmul
    double u_sum = 0.0;
    const uint8_t *u_col = up_q + (int64_t)idx * D_BPC * BLK_SZ;
    for (int b = 0; b < D_BPC; b++)
        u_sum += (double)iq2_xxs_dot_block_gpu(u_col + b * BLK_SZ, s_x + b * QK_K);

    // SiLU activation → shared memory
    float gv = (float)g_sum;
    s_act[idx] = (gv < -80.0f ? 0.0f : gv / (1.0f + expf(-gv))) * (float)u_sum;
    __syncthreads();

    // Phase 2: down matmul (D_FF → D_MODEL)
    const int FF_BPC = (D_FF + QK_K - 1) / QK_K;
    const int OUTS_PER_THREAD = 4;
    for (int o = 0; o < OUTS_PER_THREAD; o++) {
        int col = idx + o * blockDim.x;
        if (col >= D_MODEL) break;
        const uint8_t *d_col = down_q + (int64_t)col * FF_BPC * BLK_SZ;
        double d_sum = 0.0;
        for (int b = 0; b < FF_BPC; b++)
            d_sum += (double)iq2_xxs_dot_block_gpu(d_col + b * BLK_SZ, s_act + b * QK_K);
        output[col] = (float)d_sum * weight;
    }
}

// ================================================================
// Host dispatch
// ================================================================
void wubu_gpu_moe_init(void) {
    cudaMemcpyToSymbol(d_iq2xxs_grid, h_iq2xxs_grid, sizeof(h_iq2xxs_grid));
    cudaMemcpyToSymbol(d_ksigns_iq2xs, h_ksigns_iq2xs, sizeof(h_ksigns_iq2xs));
    cudaMemcpyToSymbol(d_kmask_iq2xs, h_kmask_iq2xs, sizeof(h_kmask_iq2xs));
}

void wubu_gpu_moe_forward_experts(
    const float *x,
    const uint8_t *const *gate_q, const int64_t gate_bytes,
    const uint8_t *const *up_q, const int64_t up_bytes,
    const uint8_t *const *down_q, const int64_t down_bytes,
    int gate_type, int up_type, int down_type,
    const float *weights,
    float *output,
    cudaStream_t stream)
{
    (void)gate_type; (void)up_type; (void)down_type;
    const int N_EXPERTS = 8;

    // Upload x to GPU once (shared across all expert launches)
    float *d_x;
    cudaMallocAsync(&d_x, D_MODEL * sizeof(float), stream);
    cudaMemcpyAsync(d_x, x, D_MODEL * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Allocate per-expert output on GPU
    float *d_out;
    cudaMallocAsync((void**)&d_out, D_MODEL * sizeof(float), stream);

    size_t smem_bytes = (D_FF + D_MODEL) * sizeof(float);

    // Launch one kernel per expert
    for (int e = 0; e < N_EXPERTS; e++) {
        if (weights[e] < 1e-30f) continue;  // skip zero-weight experts

        float *d_gate, *d_up, *d_down;
        cudaMallocAsync((void**)&d_gate, (size_t)gate_bytes, stream);
        cudaMallocAsync((void**)&d_up,   (size_t)up_bytes, stream);
        cudaMallocAsync((void**)&d_down, (size_t)down_bytes, stream);

        cudaMemcpyAsync(d_gate, gate_q[e], (size_t)gate_bytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_up,   up_q[e],   (size_t)up_bytes,   cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_down, down_q[e], (size_t)down_bytes, cudaMemcpyHostToDevice, stream);

        // Zero output buffer
        cudaMemsetAsync(d_out, 0, D_MODEL * sizeof(float), stream);

        moe_single_expert_kernel<<<1, D_FF, smem_bytes, stream>>>(
            d_x, (const uint8_t*)d_gate, (const uint8_t*)d_up, (const uint8_t*)d_down, weights[e], d_out);

        // Download
        cudaMemcpyAsync(output + (int64_t)e * D_MODEL, d_out,
                        D_MODEL * sizeof(float), cudaMemcpyDeviceToHost, stream);

        cudaFreeAsync(d_gate, stream);
        cudaFreeAsync(d_up, stream);
        cudaFreeAsync(d_down, stream);
    }

    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_x, stream);
    cudaFreeAsync(d_out, stream);
}
