/**
 * gpu_output_proj.cu — cuBLAS-accelerated output projection for gen_text.
 *
 * The output projection (2048 × 248320) is the single largest matmul.
 * On CPU it takes ~11ms per decode step. On GPU (RTX 5050) it takes ~0.1ms.
 *
 * Strategy:
 * 1. At init: dequant output.weight Q4_K → F32, upload to GPU, keep resident
 * 2. Per decode step: upload hidden_state [2048], cuBLAS SGEMM, download logits [248320]
 * 3. No per-step dequant or weight transfer needed
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "gguf_reader.h"
#include "wubu_ssm.h"

// GPU resources (persistent across decode steps)
static cublasHandle_t g_cublas = NULL;
static cudaStream_t   g_stream = NULL;
static float         *g_d_weight = NULL;  // [V, D] F32 output weight on GPU
static float         *g_d_x = NULL;       // [1, D] input hidden state
static float         *g_d_y = NULL;       // [1, V] output logits
static int g_vocab_size = 0;
static int g_initialized = 0;

// Dequant Q4_K block to F32 (inline for simplicity)
// block_q4_K: d(uint16), dmin(uint16), scales[12], qs[128]
// Each Q4_K block encodes 256 values in 128 bytes (4-bit each)
static void dequant_block_q4_K_f32(const uint8_t *block, float *out) {
    const uint16_t d_half = *(const uint16_t*)(block);
    const uint16_t dmin_half = *(const uint16_t*)(block + 2);
    // Convert fp16 to fp32 (simple conversion)
    int sign = (d_half >> 15) & 1;
    int exp  = (d_half >> 10) & 0x1F;
    int mant = d_half & 0x3FF;
    float d;
    if (exp == 0) d = (float)mant / 1024.0f * 0.000061035f * (sign ? -1.0f : 1.0f);
    else if (exp == 31) d = sign ? -__builtin_huge_valf() : __builtin_huge_valf();
    else d = ldexpf(1.0f + (float)mant / 1024.0f, exp - 15) * (sign ? -1.0f : 1.0f);

    sign = (dmin_half >> 15) & 1;
    exp  = (dmin_half >> 10) & 0x1F;
    mant = dmin_half & 0x3FF;
    float dmin;
    if (exp == 0) dmin = (float)mant / 1024.0f * 0.000061035f * (sign ? -1.0f : 1.0f);
    else if (exp == 31) dmin = sign ? -__builtin_huge_valf() : __builtin_huge_valf();
    else dmin = ldexpf(1.0f + (float)mant / 1024.0f, exp - 15) * (sign ? -1.0f : 1.0f);

    // Decode scales (simplified - same as decode_scales in quantized_dot_generic.c)
    const uint8_t *scales_raw = block + 4;
    uint32_t utmp[4];
    memcpy(utmp, scales_raw, 12);
    utmp[3] = ((utmp[2] >> 4) & 0x0f0f0f0f) | (((utmp[1] >> 6) & 0x03030303) << 4);
    const uint32_t uaux = utmp[1] & 0x3f3f3f3f;
    utmp[1] = (utmp[2] & 0x0f0f0f0f) | (((utmp[0] >> 6) & 0x03030303) << 4);
    utmp[2] = uaux;
    utmp[0] &= 0x3f3f3f3f;
    const uint8_t *scales = (const uint8_t*)&utmp[0];
    const uint8_t *mins   = (const uint8_t*)&utmp[2];

    const uint8_t *qs = block + 4 + 12;  // qs starts after scales
    int idx = 0;
    for (int j = 0; j < 8; j++) {  // 8 sub-blocks of 32
        float sc = scales[j];
        float mn = mins[j/2];
        for (int k = 0; k < 32; k += 2) {
            int v0 = qs[idx/2] & 0xF;
            int v1 = qs[idx/2] >> 4;
            out[idx] = (v0 * sc - mn) * d;
            out[idx+1] = (v1 * sc - mn) * d;
            idx += 2;
        }
        if (j % 2 == 1) qs += 16;  // each pair of sub-blocks uses 16 bytes
    }
}

// Initialize GPU output projection
// Dequants output weight Q4_K → F32 on CPU, uploads to GPU
bool gpu_output_init(const uint8_t *weight_q, int D, int V, int weight_type) {
    if (g_initialized) return true;
    
    if (weight_type != GGML_TYPE_Q4_K) {
        fprintf(stderr, "GPU output proj: expected Q4_K, got %d\n", weight_type);
        return false;
    }

    cudaError_t ce;
    cublasStatus_t cs;

    // Create CUDA stream and cuBLAS handle
    ce = cudaStreamCreate(&g_stream);
    if (ce != cudaSuccess) { fprintf(stderr, "cudaStreamCreate: %s\n", cudaGetErrorString(ce)); return false; }
    cs = cublasCreate(&g_cublas);
    if (cs != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasCreate failed\n"); return false; }
    cs = cublasSetStream(g_cublas, g_stream);
    if (cs != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasSetStream failed\n"); return false; }

    // Dequant Q4_K → F32 on CPU
    // Q4_K block size: 256 values per block
    const int QK_K = 256;
    const int Q4K_BLOCK_BYTES = 4 + 12 + 128;  // d+dmin(4) + scales(12) + qs(128) = 144
    // Actually: sizeof(block_q4_K) = 2+2+12+128 = 144 bytes
    const int blk_sz_q4k = 144;
    int n_blocks = (D * V + QK_K - 1) / QK_K;
    
    float *weight_f32 = (float*)malloc((size_t)D * V * sizeof(float));
    if (!weight_f32) { fprintf(stderr, "Failed to allocate F32 weight (%zu bytes)\n", (size_t)D * V * 4); return false; }
    
    // Dequant block by block
    // Weight layout: [D, V] = [2048, 248320], V is the fast dimension
    // Q4_K blocks are stored contiguously along the V dimension
    // n_blocks_per_col = (V + QK_K - 1) / QK_K  -- but V=248320, QK_K=256, so 970 blocks per row
    for (int i = 0; i < D; i++) {
        const uint8_t *row_start = weight_q + i * (n_blocks / D) * blk_sz_q4k;
        // Actually, the layout is: weight[D, V] quantized as Q4_K blocks along V
        // For each row of D: V elements = ceil(V/256) Q4_K blocks
        int blocks_per_row = (V + QK_K - 1) / QK_K;  // 970
        for (int b = 0; b < blocks_per_row; b++) {
            const uint8_t *block = row_start + b * blk_sz_q4k;
            float block_out[QK_K];
            dequant_block_q4_K_f32(block, block_out);
            int n_vals = (b == blocks_per_row - 1) ? (V - b * QK_K) : QK_K;
            memcpy(weight_f32 + (size_t)i * V + b * QK_K, block_out, n_vals * sizeof(float));
        }
    }

    // Allocate GPU memory
    ce = cudaMalloc(&g_d_weight, (size_t)D * V * sizeof(float));
    if (ce != cudaSuccess) { free(weight_f32); fprintf(stderr, "cudaMalloc weight: %s\n", cudaGetErrorString(ce)); return false; }
    ce = cudaMalloc(&g_d_x, (size_t)D * sizeof(float));
    if (ce != cudaSuccess) { free(weight_f32); fprintf(stderr, "cudaMalloc x: %s\n", cudaGetErrorString(ce)); return false; }
    ce = cudaMalloc(&g_d_y, (size_t)V * sizeof(float));
    if (ce != cudaSuccess) { free(weight_f32); fprintf(stderr, "cudaMalloc y: %s\n", cudaGetErrorString(ce)); return false; }

    // Upload weight to GPU
    // Weight for SGEMM: output = weight^T @ x
    // weight is [D, V] in row-major, we need [V, D] for column-major cuBLAS
    // cublasSgemm(A=V, B=D): C[M,N] = A[M,K] @ B[K,N]
    // We want: logits[1, V] = x[1, D] @ weight^T[D, V]
    // Or: logits[V, 1] = weight^T[V, D] @ x[D, 1]
    // cublasSgemm with op(A)=N, op(B)=N:
    //   C[V, 1] = weight^T[V, D] @ x[D, 1]
    // But weight is stored as [D, V] in memory (row-major).
    // For cuBLAS (column-major), [D, V] row-major = [V, D] column-major = weight^T
    // So: ce = cudaMemcpy(g_d_weight, weight_f32, D*V*sizeof(float), cudaMemcpyHostToDevice);
    // Then cuBLAS call: CUBLAS_OP_N (use as-is in col-major = row-major transpose)
    
    ce = cudaMemcpy(g_d_weight, weight_f32, (size_t)D * V * sizeof(float), cudaMemcpyHostToDevice);
    free(weight_f32);
    if (ce != cudaSuccess) { fprintf(stderr, "cudaMemcpy weight: %s\n", cudaGetErrorString(ce)); return false; }

    g_vocab_size = V;
    g_initialized = 1;
    fprintf(stderr, "GPU output proj: initialized (%d×%d, %.1f MB)\n", D, V, (double)D*V*4/1048576);
    return true;
}

// Run output projection on GPU
// input: [D_MODEL] float32 hidden state
// output: [vocab_size] float32 logits
bool gpu_output_project(const float *input, float *output) {
    if (!g_initialized) { fprintf(stderr, "GPU not initialized\n"); return false; }
    
    int D = D_MODEL;
    int V = g_vocab_size;
    
    cudaError_t ce;
    
    // Upload input
    ce = cudaMemcpyAsync(g_d_x, input, D * sizeof(float), cudaMemcpyHostToDevice, g_stream);
    if (ce != cudaSuccess) { fprintf(stderr, "cudaMemcpyAsync input: %s\n", cudaGetErrorString(ce)); return false; }
    
    // cuBLAS SGEMM: y[V] = weight^T[V, D] @ x[D]
    // weight layout in memory: [D, V] row-major
    // cuBLAS expects column-major, so [D, V] row-major = [V, D] column-major = weight^T
    // So we can use weight as-is with CUBLAS_OP_N
    // C = op(A) @ op(B), C[M,N], A[M,K], B[K,N]
    // We want: y[V,1] = weight^T[V,D] @ x[D,1]
    // A = weight: [V, D] → op(A)=N, M=V, K=D
    // B = x: [D, 1] → op(B)=N, K=D, N=1
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t cs = cublasSgemm(g_cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        V, 1, D,
        &alpha,
        g_d_weight, V,
        g_d_x, D,
        &beta,
        g_d_y, V);
    if (cs != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasSgemm failed\n"); return false; }
    
    // Download result
    ce = cudaMemcpyAsync(output, g_d_y, V * sizeof(float), cudaMemcpyDeviceToHost, g_stream);
    if (ce != cudaSuccess) { fprintf(stderr, "cudaMemcpyAsync output: %s\n", cudaGetErrorString(ce)); return false; }
    
    cudaStreamSynchronize(g_stream);
    return true;
}

// Cleanup
void gpu_output_cleanup() {
    if (g_d_weight) cudaFree(g_d_weight);
    if (g_d_x) cudaFree(g_d_x);
    if (g_d_y) cudaFree(g_d_y);
    if (g_cublas) cublasDestroy(g_cublas);
    if (g_stream) cudaStreamDestroy(g_stream);
    g_initialized = 0;
    fprintf(stderr, "GPU output proj: cleaned up\n");
}
