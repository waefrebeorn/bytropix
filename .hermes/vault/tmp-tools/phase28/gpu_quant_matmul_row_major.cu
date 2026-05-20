/**
 * gpu_quant_matmul.cu — GPU quantized matmul kernels for Q5_K and Q6_K.
 *
 * ORIGINAL KERNELS assume COLUMN-MAJOR weight layout (stride blocks = ceil(n_rows/QK_K))
 * but GGUF stores ROW-MAJOR [D_MODEL, CONV_DIM]. These kernels read WRONG data.
 *
 * NEW KERNELS (row_major variant): each thread handles one input ROW instead of one output column.
 * Thread loads all 32 blocks for its row (contiguous), dequantizes, atomically accumulates.
 */

#include "gpu_quant_matmul.h"
#include "gguf_reader.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define QK_K 256

// Shared helpers
static __device__ float fp16_to_fp32_dev(uint16_t h) {
    // F16: sign=1, exp=5, mant=10 → F32: sign=1, exp=8, mant=23
    uint32_t s = (h & 0x8000) << 16;
    uint32_t e = (h & 0x7C00) >> 10;
    uint32_t m = (h & 0x03FF);
    if (e == 0) {  // zero/subnormal
        if (m == 0) return *((float*)&s);  // zero
        // Subnormal → normalize
        uint32_t m_norm = m;
        while (!(m_norm & 0x0400)) { m_norm <<= 1; e--; }
        e++; m_norm &= 0x03FF;
    } else if (e == 31) {  // inf/nan
        e = 255;
    } else {
        e += 127 - 15;
    }
    uint32_t bits = s | (e << 23) | (m << 13);
    return *((float*)&bits);
}

// get_scale_min_k4_dev from original kernel
static __device__ void get_scale_min_k4_dev(int j, const uint8_t *sc, uint8_t *s, uint8_t *m) {
    if (j < 4) {
        *s = sc[j] & 0x3F;
        *m = sc[j] >> 6;
    } else {
        *s = (sc[j+4] & 0xF) | ((sc[j-4] >> 6) << 4);
        *m = (sc[j+4] >> 4) | ((sc[j] >> 6) << 4);
    }
}

// ================================================================
// Q5_K Matmul — ROW-MAJOR version (correct for GGUF layout)
// Each thread handles one input ROW: loads 32 contiguous blocks,
// dequantizes 256 elements at a time, atomically accumulates.
// ================================================================
__global__ void quant_matmul_q5_k_row_major(const float *x, const uint8_t *W_q,
                                            float *y, int n_rows, int n_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    
    const int blocks_per_row = n_cols / QK_K;  // 32 for 8192 cols
    const uint8_t *row_data = W_q + (int64_t)row * blocks_per_row * 176;
    
    for (int s = 0; s < blocks_per_row; s++) {
        const uint8_t *block = row_data + (int64_t)s * 176;
        float d = fp16_to_fp32_dev(*(const uint16_t *)block);
        float dmin = fp16_to_fp32_dev(*(const uint16_t *)(block + 2));
        const uint8_t *sc = block + 4, *qh = block + 16, *qs = block + 48;
        int col_base = s * QK_K;  // 256 columns per block
        
        // Process 256 elements in groups of 64 (original kernel pattern)
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t s1, m1, s2, m2;
            get_scale_min_k4_dev(is+0, sc, &s1, &m1);
            get_scale_min_k4_dev(is+1, sc, &s2, &m2);
            float d1 = d*s1, ml1 = dmin*m1, d2 = d*s2, ml2 = dmin*m2;
            int qb = j/2, ci = j/64;
            for (int l = 0; l < 32; l++) {
                uint8_t lo = qs[qb + l];
                uint8_t hi0 = (qh[l] >> (ci*2 + 0)) & 1;
                uint8_t hi1 = (qh[l] >> (ci*2 + 1)) & 1;
                float v0 = d1 * ((lo & 0x0F) + (hi0 ? 16 : 0)) - ml1;
                atomicAdd(&y[col_base + j + l], x[row] * v0);
                float v1 = d2 * ((lo >> 4) + (hi1 ? 16 : 0)) - ml2;
                atomicAdd(&y[col_base + j + 32 + l], x[row] * v1);
            }
            is += 2;
        }
    }
}

// ================================================================
// Q6_K Matmul — ROW-MAJOR version (correct for GGUF layout)
// Same structure: each thread handles one input row.
// ================================================================
__global__ void quant_matmul_q6_k_row_major(const float *x, const uint8_t *W_q,
                                            float *y, int n_rows, int n_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    
    const int blocks_per_row = n_cols / QK_K;  // 8 for 2048 cols (ssm_out is [4096,2048])
    const uint8_t *row_data = W_q + (int64_t)row * blocks_per_row * 210;
    
    for (int s = 0; s < blocks_per_row; s++) {
        const uint8_t *block = row_data + (int64_t)s * 210;
        float d = fp16_to_fp32_dev(*(const uint16_t *)(block + 208));
        const uint8_t *ql = block, *qh = block + 128;
        const int8_t *sc = (const int8_t *)(block + 192);
        int col_base = s * QK_K;
        
        for (int ni = 0; ni < QK_K; ni += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int idx0 = col_base + ni + l;
                int idx1 = col_base + ni + 32 + l;
                int idx2 = col_base + ni + 64 + l;
                int idx3 = col_base + ni + 96 + l;
                uint8_t l0 = ql[l + 0];
                uint8_t l32 = ql[l + 32];
                uint8_t h = qh[l];
                
                int8_t v6_0 = (int8_t)((l0 & 0xF) | ((h >> 0) & 3) << 4);
                atomicAdd(&y[idx0], x[row] * ((float)d * sc[is+0] * v6_0 - 32.0f));
                
                int8_t v6_1 = (int8_t)((l32 & 0xF) | ((h >> 2) & 3) << 4);
                atomicAdd(&y[idx1], x[row] * ((float)d * sc[is+2] * v6_1 - 32.0f));
                
                int8_t v6_2 = (int8_t)((l0 >> 4) | ((h >> 4) & 3) << 4);
                atomicAdd(&y[idx2], x[row] * ((float)d * sc[is+4] * v6_2 - 32.0f));
                
                int8_t v6_3 = (int8_t)((l32 >> 4) | ((h >> 6) & 3) << 4);
                atomicAdd(&y[idx3], x[row] * ((float)d * sc[is+6] * v6_3 - 32.0f));
            }
            ql += 64; qh += 32; sc += 8;
        }
    }
}

// ================================================================
// Wrapper — row-major aware dispatcher
// ================================================================
extern "C"
int wubu_cuda_quant_matmul_row_major(const float *x, const uint8_t *W_q, int qt,
    int nr, int nc, float *y, cudaStream_t st) {
    // nr = n_rows (D_MODEL), nc = n_cols (CONV_DIM or VALUE_DIM)
    int block = 256;
    int grid = (nr + block - 1) / block;  // one thread per input row
    switch (qt) {
    case GGML_TYPE_Q5_K:
        quant_matmul_q5_k_row_major<<<grid, block, 0, st>>>(x, W_q, y, nr, nc);
        return 1;
    case GGML_TYPE_Q6_K:
        quant_matmul_q6_k_row_major<<<grid, block, 0, st>>>(x, W_q, y, nr, nc);
        return 1;
    default:
        return 0;
    }
}
