/**
 * gpu_moe_kernel.cu — GPU MoE expert compute for IQ2_XXS/IQ3_XXS quantized weights.
 *
 * v3: Per-expert kernel launches with pre-allocated GPU buffers.
 * No cudaMalloc/cudaFree per expert (saves ~320 mallocs/frees per decode).
 */
#include "gpu_moe_kernel.h"
#include "gguf_reader.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define QK_K 256
#define D_FF 512
#define D_MODEL 2048

static __constant__ uint64_t d_iq2xxs_grid[256];
static __constant__ uint8_t  d_ksigns_iq2xs[128];
static __constant__ uint8_t  d_kmask_iq2xs[8];
static const uint8_t h_kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

#include "iq2xxs_grid_data.inc"

static __device__ __inline__ float d_f16_f32(uint16_t v) {
    int s=(v>>15)&1,e=(v>>10)&0x1F,m=v&0x3FF; uint32_t f;
    if(e==0)f=(s<<31);else if(e==31)f=(s<<31)|(0xFF<<23)|(m<<13);
    else f=(s<<31)|((127-15+e)<<23)|(m<<13);
    float r; __builtin_memcpy(&r,&f,4); return r;
}

static __device__ float iq2_xxs_dot(const uint8_t *block, const float *x) {
    float d = d_f16_f32(*(const uint16_t *)block);
    const uint16_t *qs16 = (const uint16_t *)(block + 2);
    float total = 0.0f;
    for (int ib32 = 0; ib32 < QK_K/32; ib32++) {
        uint32_t aux32[2];
        const uint8_t *src = (const uint8_t*)(qs16 + 4*ib32);
        aux32[0]=((uint32_t)src[0])|(((uint32_t)src[1])<<8)|(((uint32_t)src[2])<<16)|(((uint32_t)src[3])<<24);
        aux32[1]=((uint32_t)src[4])|(((uint32_t)src[5])<<8)|(((uint32_t)src[6])<<16)|(((uint32_t)src[7])<<24);
        const uint8_t *aux8 = (const uint8_t *)aux32;
        float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
        for (int l = 0; l < 4; l++) {
            const uint8_t *grid = (const uint8_t*)(&d_iq2xxs_grid[aux8[l]]);
            uint8_t signs = d_ksigns_iq2xs[(aux32[1] >> (7*l)) & 127];
            int base = ib32 * 32 + l * 8;
            for (int j = 0; j < 8; j++) {
                float val = db * (float)grid[j];
                if (signs & d_kmask_iq2xs[j]) total -= x[base + j] * val;
                else total += x[base + j] * val;
            }
        }
    }
    return total;
}

// Per-expert kernel: 1 block × 512 threads, 10KB shared mem
__global__ void moe_expert_kernel(
    const float * __restrict__ x,
    const uint8_t * __restrict__ gate_q,
    const uint8_t * __restrict__ up_q,
    const uint8_t * __restrict__ down_q,
    float weight,
    float * __restrict__ output)
{
    extern __shared__ float smem[];
    float *s_act = smem;
    float *s_x   = smem + D_FF;
    int idx = threadIdx.x;

    for (int i = idx; i < D_MODEL; i += blockDim.x) s_x[i] = x[i];
    __syncthreads();

    const int D_BPC = (D_MODEL + QK_K - 1) / QK_K;
    const int FF_BPC = (D_FF + QK_K - 1) / QK_K;
    const int BLK_SZ = 66;
    const int OUTS = 4;

    double gs = 0.0, us = 0.0;
    const uint8_t *gc = gate_q + (int64_t)idx * D_BPC * BLK_SZ;
    const uint8_t *uc = up_q   + (int64_t)idx * D_BPC * BLK_SZ;
    for (int b = 0; b < D_BPC; b++) {
        gs += (double)iq2_xxs_dot(gc + b * BLK_SZ, s_x + b * QK_K);
        us += (double)iq2_xxs_dot(uc + b * BLK_SZ, s_x + b * QK_K);
    }
    float gv = (float)gs;
    s_act[idx] = (gv < -80.0f ? 0.0f : gv / (1.0f + expf(-gv))) * (float)us;
    __syncthreads();

    for (int o = 0; o < OUTS; o++) {
        int col = idx + o * blockDim.x;
        if (col >= D_MODEL) break;
        const uint8_t *dc = down_q + (int64_t)col * FF_BPC * BLK_SZ;
        double ds = 0.0;
        for (int b = 0; b < FF_BPC; b++)
            ds += (double)iq2_xxs_dot(dc + b * BLK_SZ, s_act + b * QK_K);
        output[col] = (float)ds * weight;
    }
}

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
    cudaStream_t stream,
    uint8_t *d_gate_buf, uint8_t *d_up_buf, uint8_t *d_down_buf,
    float *d_x_buf,          // [D_MODEL] pre-allocated input buffer
    float *d_out_buf, float *d_weights_buf,
    bool use_gpu_ptrs)       // true: gate_q/up_q/down_q are GPU pointers
{
    (void)gate_type; (void)up_type; (void)down_type;
    (void)d_weights_buf;
    const int N_EXPERTS = 8;
    const size_t smem_bytes = (D_FF + D_MODEL) * sizeof(float);

    // Upload x once (to pre-allocated buffer)
    cudaMemcpyAsync(d_x_buf, x, D_MODEL * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Per-expert: get weights (H2D upload or direct GPU pointers), launch kernel
    for (int e = 0; e < N_EXPERTS; e++) {
        if (weights[e] < 1e-30f) {
            memset(output + (int64_t)e * D_MODEL, 0, D_MODEL * sizeof(float));
            continue;
        }
        // Get the weight data pointers on GPU
        const uint8_t *d_gate, *d_up, *d_down;
        if (use_gpu_ptrs) {
            // Pointers are already on GPU (cache hit) — use directly
            d_gate = gate_q[e];
            d_up   = up_q[e];
            d_down = down_q[e];
        } else {
            // Host pointers — upload to scratch buffers first
            cudaMemcpyAsync(d_gate_buf, gate_q[e], (size_t)gate_bytes, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_up_buf,   up_q[e],   (size_t)up_bytes,   cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_down_buf, down_q[e], (size_t)down_bytes, cudaMemcpyHostToDevice, stream);
            d_gate = d_gate_buf;
            d_up   = d_up_buf;
            d_down = d_down_buf;
        }

        moe_expert_kernel<<<1, D_FF, smem_bytes, stream>>>(
            d_x_buf, d_gate, d_up, d_down, weights[e], d_out_buf);

        cudaMemcpyAsync(output + (int64_t)e * D_MODEL, d_out_buf,
                        D_MODEL * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }

    cudaStreamSynchronize(stream);
}
