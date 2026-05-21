/**
 * gpu_moe_kernel.cu — GPU MoE expert compute with Q8_K input quantization.
 *
 * v5: Q8_K quantize x before dot product to match CPU quantized_matmul exactly.
 *     Uses integer arithmetic (int8 grid × int8 q8) like CPU vec_dot.
 *     extern __shared__ float only (no static __shared__ — avoids sm_120 compiler bug).
 *     Thread 0 does between-warps reduction (avoids sm_120 __syncthreads-in-loop bug).
 */
#include "gpu_moe_kernel.h"
#include "gguf_reader.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define QK_K 256
#define D_FF 512
#define D_MODEL 2048
#define WARP_SIZE 32

static __constant__ uint64_t d_iq2xxs_grid[256];
static __constant__ uint8_t  d_ksigns_iq2xs[128];
static __constant__ uint8_t  d_kmask_iq2xs[8];
static __constant__ uint32_t d_iq3xxs_grid[256];
static const uint8_t h_kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

#include "iq2xxs_grid_data.inc"

static const uint32_t h_iq3xxs_grid[256] = {
#include "iq3xxs_grid.inc"
};

static __device__ __inline__ float d_f16_f32(uint16_t v) {
    int s=(v>>15)&1,e=(v>>10)&0x1F,m=v&0x3FF;
    if(e==0){ float r = (float)m * 0x1p-24f; return s ? -r : r; }
    else if(e==31){uint32_t f=(s<<31)|(0xFF<<23)|(m<<13);float r;__builtin_memcpy(&r,&f,4);return r;}
    else{uint32_t f=(s<<31)|((127-15+e)<<23)|(m<<13);float r;__builtin_memcpy(&r,&f,4);return r;}
}

// ============================================================
// IQ2_XXS dot product with Q8_K input (matches CPU vec_dot)
// Returns: d_iq * bsum * 0.125  (caller multiplies by d_q8)
// ============================================================
static __device__ float iq2_xxs_dot_q8(const uint8_t *iq, const int8_t *q8) {
    float d_iq = d_f16_f32(*(const uint16_t *)iq);
    const uint16_t *qs16 = (const uint16_t *)(iq + 2);
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < QK_K/32; ib32++) {
        uint32_t aux32[2];
        const uint8_t *src = (const uint8_t*)(qs16 + 4*ib32);
        aux32[0]=src[0]|(src[1]<<8)|(src[2]<<16)|(src[3]<<24);
        aux32[1]=src[4]|(src[5]<<8)|(src[6]<<16)|(src[7]<<24);
        uint32_t ls = 2*(aux32[1] >> 28) + 1;
        const uint8_t *aux8 = (const uint8_t *)aux32;
        int32_t sumi = 0;
        for (int l = 0; l < 4; l++) {
            const uint8_t *grid = (const uint8_t*)(&d_iq2xxs_grid[aux8[l]]);
            uint8_t signs = d_ksigns_iq2xs[(aux32[1] >> (7*l)) & 127];
            int base = ib32 * 32 + l * 8;
            for (int j = 0; j < 8; j++) {
                int s = (signs & d_kmask_iq2xs[j]) ? -1 : 1;
                sumi += (int32_t)grid[j] * (int32_t)q8[base + j] * s;
            }
        }
        bsum += sumi * (int32_t)ls;
    }
    return d_iq * (float)bsum * 0.125f;
}

static __device__ float iq3_xxs_dot_q8(const uint8_t *iq, const int8_t *q8) {
    float d_iq = d_f16_f32(*(const uint16_t *)iq);
    const uint8_t *qs = iq + 2;
    const uint8_t *ss = qs + 64;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < QK_K/32; ib32++) {
        const uint8_t *s = ss + 4*ib32;
        uint32_t aux32 = s[0]|(s[1]<<8)|(s[2]<<16)|(s[3]<<24);
        uint32_t ls = 2*(aux32 >> 28) + 1;
        const uint8_t *q = qs + ib32 * 8;
        int32_t sumi = 0;
        for (int l = 0; l < 4; l++) {
            uint8_t signs = d_ksigns_iq2xs[(aux32 >> (7*l)) & 127];
            uint8_t g1 = q[2*l], g2 = q[2*l+1];
            const uint8_t *grid1 = (const uint8_t*)(&d_iq3xxs_grid[g1]);
            const uint8_t *grid2 = (const uint8_t*)(&d_iq3xxs_grid[g2]);
            int base = ib32 * 32 + l * 8;
            for (int j = 0; j < 4; j++) {
                int s1 = (signs & d_kmask_iq2xs[j]) ? -1 : 1;
                int s2 = (signs & d_kmask_iq2xs[j+4]) ? -1 : 1;
                sumi += (int32_t)(int8_t)grid1[j] * (int32_t)q8[base + j] * s1;
                sumi += (int32_t)(int8_t)grid2[j] * (int32_t)q8[base + j + 4] * s2;
            }
        }
        bsum += sumi * (int32_t)ls;
    }
    return d_iq * (float)bsum * 0.125f;
}

static __device__ float dot_q8(const uint8_t *blk, const int8_t *q8, int type) {
    if (type == GGML_TYPE_IQ2_XXS) return iq2_xxs_dot_q8(blk, q8);
    if (type == GGML_TYPE_IQ3_XXS) return iq3_xxs_dot_q8(blk, q8);
    return 0.0f;
}

static __host__ __device__ int blk_sz_for_type(int type) {
    if (type == GGML_TYPE_IQ2_XXS) return 66;
    if (type == GGML_TYPE_IQ3_XXS) return 98;
    if (type == GGML_TYPE_IQ4_XS)  return 136;
    return 66;
}

// ============================================================
// Host-side: constant init
// ============================================================
void wubu_gpu_moe_init(void) {
    cudaMemcpyToSymbol(d_iq2xxs_grid, h_iq2xxs_grid, sizeof(h_iq2xxs_grid));
    cudaMemcpyToSymbol(d_ksigns_iq2xs, h_ksigns_iq2xs, sizeof(h_ksigns_iq2xs));
    cudaMemcpyToSymbol(d_kmask_iq2xs, h_kmask_iq2xs, sizeof(h_kmask_iq2xs));
    cudaMemcpyToSymbol(d_iq3xxs_grid, h_iq3xxs_grid, sizeof(h_iq3xxs_grid));
}

// ============================================================
// Kernel: 1 block × 512 threads
// Shared memory (extern float): smem[0..D_MODEL+NW-1]
//   Phase 1: x_f32[0..D_MODEL) floats, then warp_peaks[D_MODEL..D_MODEL+NW)
//   After Phase 1: compact layout reuses x_f32 space:
//     q8_x int8[0..D_MODEL) at bytes 0..2047 (float slots 0..511)
//     d_q8_x[0..D_BPC) at float slots 512..519
//     act[0..D_FF) at float slots 520..1031
//     q8_act int8 at bytes (520+512)*4 = 4128..4639
//     d_q8_act[0..FF_BPC) at byte 4640 (float slot 1160)
//     warp_peaks[D_MODEL..D_MODEL+NW) reused (float slots 2048..2063)
// Peak: D_MODEL + NW = 2064 float slots = 8256 bytes
// ============================================================
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
    int idx = threadIdx.x;
    const int D_BPC = (D_MODEL + QK_K - 1) / QK_K;   // 8
    const int FF_BPC = (D_FF + QK_K - 1) / QK_K;      // 2
    const int OUTS = 4;
    const int NW = blockDim.x / WARP_SIZE;  // 16 warps

    // Phase 1: x_f32 = smem[0..D_MODEL), warp_peaks = smem[D_MODEL..D_MODEL+NW)
    // After Phase 1 (compact within float[0..2048)):
    int8_t *q8_x     = (int8_t *)smem;          // [D_MODEL] int8 = float slots 0..511
    float  *d_q8_x   = smem + D_MODEL/4;        // [D_BPC] = float slots 512..519
    float  *act      = d_q8_x + D_BPC;          // [D_FF] = float slots 520..1031
    int8_t *q8_act   = (int8_t *)(act + D_FF); // [D_FF] int8 at byte 4128 (slot 1032*4)
    float  *d_q8_act = (float *)(q8_act + D_FF);// [FF_BPC] float at byte 4640 (slot 1160)
    float  *warp_peaks = smem + D_MODEL;        // [NW] = float slots 2048..2063

    // === Phase 1: Copy x, quantize each block to Q8_K ===
    for (int i = idx; i < D_MODEL; i += blockDim.x) smem[i] = x[i];
    __syncthreads();

    #pragma unroll
    for (int b = 0; b < D_BPC; b++) {
        int base = b * QK_K;
        float amax = 0.0f;
        for (int i = idx; i < QK_K; i += blockDim.x) {
            float v = fabsf(smem[base + i]);
            if (v > amax) amax = v;
        }
        for (int o = WARP_SIZE/2; o > 0; o /= 2)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, o));
        if ((idx % WARP_SIZE) == 0)
            warp_peaks[idx / WARP_SIZE] = amax;
        __syncthreads();
        if (idx == 0) {
            float w = warp_peaks[0];
            #pragma unroll
            for (int k = 1; k < NW; k++)
                if (warp_peaks[k] > w) w = warp_peaks[k];
            d_q8_x[b] = fmaxf(w / 127.0f, 1e-30f);
        }
        __syncthreads();
        { float inv = 1.0f / d_q8_x[b];
          for (int i = idx; i < QK_K; i += blockDim.x) {
              q8_x[base + i] = (int8_t)rintf(smem[base + i] * inv);
          } }
        __syncthreads();
    }

    // === Phase 2: Gate/up projection using Q8_K input ===
    double gs = 0.0, us = 0.0;
    const uint8_t *gc = gate_q + (int64_t)idx * D_BPC * gate_blk_sz;
    const uint8_t *uc = up_q   + (int64_t)idx * D_BPC * up_blk_sz;
    for (int b = 0; b < D_BPC; b++) {
        gs += (double)dot_q8(gc + b * gate_blk_sz, q8_x + b * QK_K, gate_type) * d_q8_x[b];
        us += (double)dot_q8(uc + b * up_blk_sz,   q8_x + b * QK_K, up_type)   * d_q8_x[b];
    }
    float gv = (float)gs;
    act[idx] = (gv < -80.0f ? 0.0f : gv / (1.0f + expf(-gv))) * (float)us;
    __syncthreads();

    // === Phase 3: Quantize act to Q8_K ===
    #pragma unroll
    for (int b = 0; b < FF_BPC; b++) {
        int base = b * QK_K;
        float amax = 0.0f;
        for (int i = idx; i < QK_K; i += blockDim.x) {
            float v = fabsf(act[base + i]);
            if (v > amax) amax = v;
        }
        for (int o = WARP_SIZE/2; o > 0; o /= 2)
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, o));
        if ((idx % WARP_SIZE) == 0)
            warp_peaks[idx / WARP_SIZE] = amax;
        __syncthreads();
        if (idx == 0) {
            float w = warp_peaks[0];
            #pragma unroll
            for (int k = 1; k < NW; k++)
                if (warp_peaks[k] > w) w = warp_peaks[k];
            d_q8_act[b] = fmaxf(w / 127.0f, 1e-30f);
        }
        __syncthreads();
        { float inv = 1.0f / d_q8_act[b];
          for (int i = idx; i < QK_K; i += blockDim.x) {
              q8_act[base + i] = (int8_t)rintf(act[base + i] * inv);
          } }
        __syncthreads();
    }

    // === Phase 4: Down projection ===
    for (int o = 0; o < OUTS; o++) {
        int col = idx + o * blockDim.x;
        if (col >= D_MODEL) break;
        const uint8_t *dc = down_q + (int64_t)col * FF_BPC * down_blk_sz;
        double ds = 0.0;
        for (int b = 0; b < FF_BPC; b++)
            ds += (double)dot_q8(dc + b * down_blk_sz, q8_act + b * QK_K, down_type) * d_q8_act[b];
        output[col] = (float)ds * weight;
    }
}

// ============================================================
// Host-side forward: upload x, loop over experts, launch kernel
// ============================================================
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
    float *d_x_buf,
    float *d_out_buf, float *d_weights_buf,
    bool use_gpu_ptrs)
{
    (void)d_weights_buf;
    const int N_EXPERTS = 8;
    const size_t smem_bytes = (D_MODEL + 16) * sizeof(float);

    int gate_blk_sz = blk_sz_for_type(gate_type);
    int up_blk_sz   = blk_sz_for_type(up_type);
    int down_blk_sz = blk_sz_for_type(down_type);

    cudaMemcpyAsync(d_x_buf, x, D_MODEL * sizeof(float), cudaMemcpyHostToDevice, stream);

    for (int e = 0; e < N_EXPERTS; e++) {
        if (weights[e] < 1e-30f) {
            memset(output + (int64_t)e * D_MODEL, 0, D_MODEL * sizeof(float));
            continue;
        }
        const uint8_t *d_gate, *d_up, *d_down;
        if (use_gpu_ptrs) {
            d_gate = gate_q[e];
            d_up   = up_q[e];
            d_down = down_q[e];
        } else {
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
