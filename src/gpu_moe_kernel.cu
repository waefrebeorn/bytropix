/**
 * gpu_moe_kernel.cu — GPU MoE expert compute for IQ2_XXS/IQ3_XXS quantized weights.
 *
 * v4: Multi-type dequant support. Gate/up = IQ2_XXS (or type passed).
 * Down = IQ2_XXS/IQ3_XXS/IQ4_XS, dispatched by type param at kernel launch.
 * Pre-allocated GPU buffers avoid cudaMalloc/cudaFree per expert.
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
static __constant__ uint32_t d_iq3xxs_grid[256];
static const uint8_t h_kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

#include "iq2xxs_grid_data.inc"

// iq3xxs_grid: 256 entries, each uint32 packs 4 int8 values
static const uint32_t h_iq3xxs_grid[256] = {
#include "iq3xxs_grid.inc"
};

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

/**
 * IQ3_XXS (3.0625 bpw) GPU dot product.
 * Block layout: d[F16:2] + qs[64] + scales_and_signs[32] = 98 bytes
 *   qs: 64 uint8 grid indices (64 × 8-bit → 256 positions = QK_K)
 *   scales_and_signs: 8 × uint32, each:
 *     upper 4 bits: scale nibble (db = d*(0.5+scale)*0.5)
 *     lower 28 bits: 4 × 7-bit sign indices → ksigns_iq2xs lookup
 * Per ib32: 4 groups × 8 elems = 32 elements; 8 ib32 × 32 = 256 total
 */
static __device__ float iq3_xxs_dot(const uint8_t *block, const float *x) {
    float d = d_f16_f32(*(const uint16_t *)block);
    const uint8_t *qs = block + 2;              // 64 bytes grid indices
    const uint8_t *scales_and_signs = qs + 64;  // 32 bytes
    float total = 0.0f;
    for (int ib32 = 0; ib32 < QK_K/32; ib32++) {
        // Manual byte load for unaligned access (scales_and_signs offset = 66, not 4-aligned)
        const uint8_t *ss_src = scales_and_signs + 4*ib32;
        uint32_t aux32 = ((uint32_t)ss_src[0]) | (((uint32_t)ss_src[1])<<8)
                       | (((uint32_t)ss_src[2])<<16) | (((uint32_t)ss_src[3])<<24);
        float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        const uint8_t *qs_batch = qs + ib32 * 8;
        for (int l = 0; l < 4; l++) {
            uint8_t signs = d_ksigns_iq2xs[(aux32 >> (7*l)) & 127];
            uint8_t idx1 = qs_batch[2*l+0];
            uint8_t idx2 = qs_batch[2*l+1];
            const uint8_t *grid1 = (const uint8_t *)(&d_iq3xxs_grid[idx1]);
            const uint8_t *grid2 = (const uint8_t *)(&d_iq3xxs_grid[idx2]);
            int base = ib32 * 32 + l * 8;
            for (int j = 0; j < 4; j++) {
                float v1 = db * (float)(int8_t)grid1[j];
                float v2 = db * (float)(int8_t)grid2[j];
                if (signs & d_kmask_iq2xs[j+0]) total -= x[base + j+0] * v1;
                else total += x[base + j+0] * v1;
                if (signs & d_kmask_iq2xs[j+4]) total -= x[base + j+4] * v2;
                else total += x[base + j+4] * v2;
            }
        }
    }
    return total;
}

/** Block size for a GGML quant type (only IQ2_XXS/IQ3_XXS/IQ4_XS supported). */
static __host__ __device__ int blk_sz_for_type(int type) {
    if (type == GGML_TYPE_IQ2_XXS) return 66;
    if (type == GGML_TYPE_IQ3_XXS) return 98;
    if (type == GGML_TYPE_IQ4_XS)  return 136;
    return 66; // default to IQ2_XXS
}

/** Dot product dispatching to correct IQ dequant function by type. */
static __device__ float dot_by_type(const uint8_t *block, const float *x, int type) {
    if (type == GGML_TYPE_IQ2_XXS) return iq2_xxs_dot(block, x);
    if (type == GGML_TYPE_IQ3_XXS) return iq3_xxs_dot(block, x);
    // IQ4_XS not yet implemented — return 0
    return 0.0f;
}

// Per-expert kernel: 1 block × 512 threads, 10KB shared mem
__global__ void moe_expert_kernel(
    const float * __restrict__ x,
    const uint8_t * __restrict__ gate_q,
    const uint8_t * __restrict__ up_q,
    const uint8_t * __restrict__ down_q,
    int gate_blk_sz, int up_blk_sz, int down_blk_sz,
    int gate_type, int up_type, int down_type,
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
    const int OUTS = 4;

    double gs = 0.0, us = 0.0;
    const uint8_t *gc = gate_q + (int64_t)idx * D_BPC * gate_blk_sz;
    const uint8_t *uc = up_q   + (int64_t)idx * D_BPC * up_blk_sz;
    for (int b = 0; b < D_BPC; b++) {
        gs += (double)dot_by_type(gc + b * gate_blk_sz, s_x + b * QK_K, gate_type);
        us += (double)dot_by_type(uc + b * up_blk_sz, s_x + b * QK_K, up_type);
    }
    float gv = (float)gs;
    s_act[idx] = (gv < -80.0f ? 0.0f : gv / (1.0f + expf(-gv))) * (float)us;
    __syncthreads();

    for (int o = 0; o < OUTS; o++) {
        int col = idx + o * blockDim.x;
        if (col >= D_MODEL) break;
        const uint8_t *dc = down_q + (int64_t)col * FF_BPC * down_blk_sz;
        double ds = 0.0;
        for (int b = 0; b < FF_BPC; b++)
            ds += (double)dot_by_type(dc + b * down_blk_sz, s_act + b * QK_K, down_type);
        output[col] = (float)ds * weight;
    }
}

void wubu_gpu_moe_init(void) {
    cudaMemcpyToSymbol(d_iq2xxs_grid, h_iq2xxs_grid, sizeof(h_iq2xxs_grid));
    cudaMemcpyToSymbol(d_ksigns_iq2xs, h_ksigns_iq2xs, sizeof(h_ksigns_iq2xs));
    cudaMemcpyToSymbol(d_kmask_iq2xs, h_kmask_iq2xs, sizeof(h_kmask_iq2xs));
    cudaMemcpyToSymbol(d_iq3xxs_grid, h_iq3xxs_grid, sizeof(h_iq3xxs_grid));
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
    (void)d_weights_buf;
    const int N_EXPERTS = 8;
    const size_t smem_bytes = (D_FF + D_MODEL) * sizeof(float);

    // Resolve block sizes from types
    int gate_blk_sz = blk_sz_for_type(gate_type);
    int up_blk_sz   = blk_sz_for_type(up_type);
    int down_blk_sz = blk_sz_for_type(down_type);

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
            d_x_buf, d_gate, d_up, d_down,
            gate_blk_sz, up_blk_sz, down_blk_sz,
            gate_type, up_type, down_type,
            weights[e], d_out_buf);

        cudaMemcpyAsync(output + (int64_t)e * D_MODEL, d_out_buf,
                        D_MODEL * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }

    cudaStreamSynchronize(stream);
}
