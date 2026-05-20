/**
 * gpu_quant_matmul.cu — Quantized GPU matmul for Q5_K/Q6_K types.
 *
 * Weight layout: GGUF stores weights with dims[0]=C (output dim, innermost).
 * Each column of D input-dim elements is stored as ceil(D/256) blocks.
 * y[col] = sum_r x[r] * deq(W[col][r])  — column-major weight access.
 */
#include "gpu_quant_matmul.h"
#include "gguf_reader.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define QK_K 256

static __device__ __inline__ float fp16_to_fp32_dev(uint16_t v) {
    int sign = (v >> 15) & 1, exp = (v >> 10) & 0x1F, mant = v & 0x03FF;
    uint32_t f32;
    if (exp == 0) {
        // F16 denormal: value = (-1)^sign * mant * 2^(-24).
        // Match CPU f16_to_f32() — normalize as F32 denorm+1, then subtract 2^-14.
        uint32_t norm = (sign << 31) | ((1 + 112) << 23) | (mant << 13);
        float nv; __builtin_memcpy(&nv, &norm, 4);
        return sign ? nv + 6.103515625e-5f : nv - 6.103515625e-5f;
    } else if (exp == 31) {
        f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        f32 = (sign << 31) | ((uint32_t)(127 - 15 + exp) << 23) | (mant << 13);
    }
    float r; __builtin_memcpy(&r, &f32, 4); return r;
}

static __device__ __inline__ void get_scale_min_k4_dev(int j, const uint8_t *q,
                                                       uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
           *m = (q[j+4] >> 4) | ((q[j] >> 6) << 4); }
}

// ================================================================
// Q5_K Matmul — fused dequant+dot, no bv[256] local array spill.
// y[col] = sum_r x[r] * deq(W[col][r])
// Each thread handles 1 column. Dequant in 32-element sub-blocks,
// accumulating dot product immediately (no large stack buffer).
// ================================================================
__global__ void quant_matmul_q5_k_kernel(const float *x, const uint8_t *W_q,
                                          float *y, int n_rows, int n_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_cols) return;
    const int BPC = (n_rows + QK_K - 1) / QK_K;  // blocks per column
    const uint8_t *base = W_q + (int64_t)col * BPC * 176;
    double sum = 0.0;
    for (int b = 0; b < BPC; b++) {
        const uint8_t *block = base + (int64_t)b * 176;
        float d = fp16_to_fp32_dev(*(const uint16_t *)block);
        float dmin = fp16_to_fp32_dev(*(const uint16_t *)(block + 2));
        const uint8_t *sc = block + 4, *qh = block + 16, *qs = block + 48;
        int rem = n_rows - b * QK_K;
        if (rem > QK_K) rem = QK_K;
        int is = 0;
        int x_off = b * QK_K;
        for (int j = 0; j < rem; j += 64) {
            uint8_t s1, m1, s2, m2;
            get_scale_min_k4_dev(is+0, sc, &s1, &m1);
            get_scale_min_k4_dev(is+1, sc, &s2, &m2);
            float d1 = d*s1, ml1 = dmin*m1, d2 = d*s2, ml2 = dmin*m2;
            int qb = j/2, ci = j/64;
            int lim = (j + 32 < rem) ? 32 : rem - j;
            for (int l = 0; l < lim; l++) {
                uint8_t lo = qs[qb + l];
                uint8_t hi0 = (qh[l] >> (ci*2 + 0)) & 1;
                uint8_t hi1 = (qh[l] >> (ci*2 + 1)) & 1;
                float v0 = d1 * ((lo & 0x0F) + (hi0 ? 16 : 0)) - ml1;
                sum += (double)x[x_off + j + l] * (double)v0;
                if (j + 32 + l < rem) {
                    float v1 = d2 * ((lo >> 4)   + (hi1 ? 16 : 0)) - ml2;
                    sum += (double)x[x_off + j + 32 + l] * (double)v1;
                }
            }
            is += 2;
        }
    }
    y[col] = (float)sum;
}

// ================================================================
// Q6_K Matmul — fused dequant+dot, no bv[256] local array spill.
// Each thread handles 1 column. Dequant in 32-element sub-groups,
// accumulating dot product immediately.
// ================================================================
__global__ void quant_matmul_q6_k_kernel(const float *x, const uint8_t *W_q,
                                          float *y, int n_rows, int n_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_cols) return;
    const int BPC = (n_rows + QK_K - 1) / QK_K;
    const uint8_t *base = W_q + (int64_t)col * BPC * 210;
    double sum = 0.0;
    for (int b = 0; b < BPC; b++) {
        const uint8_t *block = base + (int64_t)b * 210;
        float d = fp16_to_fp32_dev(*(const uint16_t *)(block + 208));
        const uint8_t *ql = block, *qh = block + 128;
        const int8_t *sc = (const int8_t *)(block + 192);
        int rem = n_rows - b * QK_K;
        if (rem > QK_K) rem = QK_K;
        for (int ni = 0; ni < rem; ni += 128) {
            // Process 128-element chunk in 32 iterations, each producing 4 vals
            for (int l = 0; l < 32; l++) {
                int is = l / 16;  // sc cycle: [0-3] for first half, [4-7] for second
                int idx0 = b * QK_K + ni + l;
                int idx1 = b * QK_K + ni + 32 + l;
                int idx2 = b * QK_K + ni + 64 + l;
                int idx3 = b * QK_K + ni + 96 + l;
                uint8_t l0 = ql[l + 0];
                uint8_t l32 = ql[l + 32];
                uint8_t h = qh[l];
                if (idx0 < b * QK_K + rem) {
                    int8_t v6 = (int8_t)((l0 & 0xF) | ((h >> 0) & 3) << 4);
                    sum += (double)x[idx0] * (double)d * (double)sc[is+0] * (double)(v6 - 32);
                }
                if (idx1 < b * QK_K + rem) {
                    int8_t v6 = (int8_t)((l32 & 0xF) | ((h >> 2) & 3) << 4);
                    sum += (double)x[idx1] * (double)d * (double)sc[is+2] * (double)(v6 - 32);
                }
                if (idx2 < b * QK_K + rem) {
                    int8_t v6 = (int8_t)((l0 >> 4) | ((h >> 4) & 3) << 4);
                    sum += (double)x[idx2] * (double)d * (double)sc[is+4] * (double)(v6 - 32);
                }
                if (idx3 < b * QK_K + rem) {
                    int8_t v6 = (int8_t)((l32 >> 4) | ((h >> 6) & 3) << 4);
                    sum += (double)x[idx3] * (double)d * (double)sc[is+6] * (double)(v6 - 32);
                }
            }
            ql += 64; qh += 32; sc += 8;
        }
    }
    y[col] = (float)sum;
}

// ================================================================
// Host dispatch
// ================================================================
extern "C"
size_t wubu_cuda_quant_matmul_scratch(int nr, int nc, int qt) {
    (void)nr; (void)nc; (void)qt; return 0;
}

extern "C"
int wubu_cuda_quant_matmul(const float *x, const uint8_t *W_q, int qt,
    int nr, int nc, float *y, float *scr, size_t ss, cudaStream_t st) {
    (void)scr; (void)ss;
    int block = 256;
    int grid = (nc + block - 1) / block;
    switch (qt) {
    case GGML_TYPE_Q5_K: quant_matmul_q5_k_kernel<<<grid,block,0,st>>>(x,W_q,y,nr,nc); return 1;
    case GGML_TYPE_Q6_K: quant_matmul_q6_k_kernel<<<grid,block,0,st>>>(x,W_q,y,nr,nc); return 1;
    default: return 0;
    }
}
