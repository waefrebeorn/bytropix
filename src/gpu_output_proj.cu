/**
 * gpu_output_proj.cu — cuBLAS-accelerated output projection for gen_text.
 *
 * Two modes:
 * 1. F32 mode (default): dequant Q4_K → F32, upload, cuBLAS SGEMM
 *    - Uses ~7.6GB VRAM for weight. Best for GPUs with >=8GB VRAM.
 * 2. Quantized mode (GPU_QUANTIZED=1): keep Q4_K on GPU, custom kernel
 *    - Uses ~1.9GB VRAM for weight. Best for 6.5GB VRAM laptops.
 *
 * The output projection (2048 × 248320) is the single largest matmul.
 * On CPU it takes ~10ms per decode step. On GPU (RTX 5050) it takes ~0.1ms.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#include "gguf_reader.h"
#include "wubu_ssm.h"
#include "gpu_output_proj.h"

// Q4_K block constants
#define Q4K_BLOCK_SIZE 144   // bytes per block
#define Q4K_ELEMS_PER_BLOCK 256
#define GQA_HEAD_DIM 256
#define GQA_Q_HEADS 16
#define GQA_KV_HEADS 2
#define WARP_SIZE 32

// GPU resources (persistent across decode steps)
static cublasHandle_t g_cublas = NULL;
static cudaStream_t   g_stream = NULL;
static float         *g_d_weight = NULL;  // [V, D] F32 output weight on GPU (F32 mode)
static uint8_t       *g_d_weight_q = NULL; // Q4_K weight on GPU (quantized mode)
static float         *g_d_x = NULL;       // [D_MODEL] or [D_MODEL * T] input buffer
static float         *g_d_y = NULL;       // [vocab_size] or [vocab_size * T] output buffer
static int g_vocab_size = 0;
static int g_max_batch = 0;
static int g_initialized = 0;
static int g_quantized_mode = 0;  // 1 = use custom Q4_K kernel instead of cuBLAS

// Q4_K block constants
#define Q4K_BLOCK_SIZE 144   // bytes per block
#define Q4K_ELEMS_PER_BLOCK 256

// ============================================================
// CUDA kernel: Q4_K dequant + dot product fused
//
// Each thread processes one vocabulary column:
//   y[col] = sum_i x[i] * deq(W_q[col][i])
//
// The weight W_q is [V, D] Q4_K quantized, stored contiguously.
// Each column has ceil(D/Q4K_ELEMS_PER_BLOCK) Q4_K blocks.
//
// We keep the weight on GPU in its native Q4_K format.
// ============================================================

// Device-side dequant of one Q4_K block (144 bytes → 256 floats)
__device__ static void dequant_q4_k_block_device(const uint8_t *block, float *out) {
    // Read d and dmin as fp16
    uint16_t d_bits = *(const uint16_t *)block;
    uint16_t dmin_bits = *(const uint16_t *)(block + 2);
    
    // F16 to F32 for d
    uint32_t sign = (d_bits >> 15) & 1;
    uint32_t exp  = (d_bits >> 10) & 0x1F;
    uint32_t mant = d_bits & 0x03FF;
    uint32_t f32;
    if (exp == 0) f32 = (sign << 31) | ((uint32_t)(127 - 15 + 1) << 23) | (mant << 13);
    else if (exp == 31) f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
    else f32 = (sign << 31) | ((uint32_t)(127 - 15 + exp) << 23) | (mant << 13);
    float d;
    memcpy(&d, &f32, 4);
    
    // F16 to F32 for dmin
    sign = (dmin_bits >> 15) & 1;
    exp  = (dmin_bits >> 10) & 0x1F;
    mant = dmin_bits & 0x03FF;
    if (exp == 0) f32 = (sign << 31) | ((uint32_t)(127 - 15 + 1) << 23) | (mant << 13);
    else if (exp == 31) f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
    else f32 = (sign << 31) | ((uint32_t)(127 - 15 + exp) << 23) | (mant << 13);
    float dmin;
    memcpy(&dmin, &f32, 4);
    
    const uint8_t *scales = block + 4;  // 12 bytes
    const uint8_t *qs = block + 16;     // qs starts after d+dmin+scales = 4+12 = 16
    
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        int idx = is;
        uint8_t sc1, m1, sc2, m2;
        if (idx < 4) {
            sc1 = scales[idx] & 63; m1 = scales[idx + 4] & 63;
        } else {
            sc1 = (scales[idx+4] & 0xF) | ((scales[idx-4] >> 6) << 4);
            m1  = (scales[idx+4] >>  4) | ((scales[idx  ] >> 6) << 4);
        }
        idx = is + 1;
        if (idx < 4) {
            sc2 = scales[idx] & 63; m2 = scales[idx + 4] & 63;
        } else {
            sc2 = (scales[idx+4] & 0xF) | ((scales[idx-4] >> 6) << 4);
            m2  = (scales[idx+4] >>  4) | ((scales[idx  ] >> 6) << 4);
        }
        float d1 = d * sc1; float ml1 = dmin * m1;
        float d2 = d * sc2; float ml2 = dmin * m2;
        
        const uint8_t *bq = qs + j/2;
        for (int l = 0; l < 32; l++) {
            out[j + l]           = d1 * (bq[l] & 0xF) - ml1;
            out[j + 32 + l]      = d2 * (bq[l] >> 4) - ml2;
        }
        is += 2;
    }
}

// Kernel: one thread per vocabulary column
// y[V] = x[D] @ weight_Q4_K[V, D]
// x: [D] float32
// weight_q: [V * blocks_per_col * Q4K_BLOCK_SIZE] uint8_t
// y: [V] float32 output
__global__ void quantized_output_proj_kernel(const float *x,
                                              const uint8_t *weight_q,
                                              float *y,
                                              int D, int V,
                                              int blocks_per_col) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= V) return;
    
    float sum = 0.0f;
    
    for (int b = 0; b < blocks_per_col; b++) {
        // Pointer to this block in the quantized weight
        // Weight layout: [V, D] with Q4_K blocks along D
        // Each column has `blocks_per_col` consecutive blocks
        const uint8_t *block = weight_q + (size_t)col * blocks_per_col * Q4K_BLOCK_SIZE + b * Q4K_BLOCK_SIZE;
        
        float block_vals[Q4K_ELEMS_PER_BLOCK];
        dequant_q4_k_block_device(block, block_vals);
        
        int base = b * Q4K_ELEMS_PER_BLOCK;
        int remaining = D - base;
        if (remaining > Q4K_ELEMS_PER_BLOCK) remaining = Q4K_ELEMS_PER_BLOCK;
        
        for (int i = 0; i < remaining; i++) {
            sum += x[base + i] * block_vals[i];
        }
    }
    
    y[col] = sum;
}

// ============================================================
// Host-side functions
// ============================================================

// Dequant Q4_K block to F32 (CPU version, used by F32 mode init)
static void dequant_block_q4_K_f32(const uint8_t *block, float *out) {
    uint16_t d_bits, dmin_bits;
    memcpy(&d_bits, block, 2);
    memcpy(&dmin_bits, block + 2, 2);
    int s = (d_bits >> 15) & 1, e = (d_bits >> 10) & 0x1F, m = d_bits & 0x3FF;
    float d = (e == 0) ? ldexpf((float)m / 1024.0f, -14) * (s ? -1.0f : 1.0f)
            : (e == 31) ? (s ? -__builtin_huge_valf() : __builtin_huge_valf())
            : ldexpf(1.0f + (float)m / 1024.0f, e - 15) * (s ? -1.0f : 1.0f);
    s = (dmin_bits >> 15) & 1; e = (dmin_bits >> 10) & 0x1F; m = dmin_bits & 0x3FF;
    float dmin = (e == 0) ? ldexpf((float)m / 1024.0f, -14) * (s ? -1.0f : 1.0f)
               : (e == 31) ? (s ? -__builtin_huge_valf() : __builtin_huge_valf())
               : ldexpf(1.0f + (float)m / 1024.0f, e - 15) * (s ? -1.0f : 1.0f);

    const uint8_t *scales = block + 4;
    const uint8_t *qs = block + 16;

    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        int idx = is;
        uint8_t sc1, m1, sc2, m2;
        if (idx < 4) {
            sc1 = scales[idx] & 63; m1 = scales[idx + 4] & 63;
        } else {
            sc1 = (scales[idx+4] & 0xF) | ((scales[idx-4] >> 6) << 4);
            m1  = (scales[idx+4] >>  4) | ((scales[idx  ] >> 6) << 4);
        }
        idx = is + 1;
        if (idx < 4) {
            sc2 = scales[idx] & 63; m2 = scales[idx + 4] & 63;
        } else {
            sc2 = (scales[idx+4] & 0xF) | ((scales[idx-4] >> 6) << 4);
            m2  = (scales[idx+4] >>  4) | ((scales[idx  ] >> 6) << 4);
        }
        float d1 = d * sc1; float ml1 = dmin * m1;
        float d2 = d * sc2; float ml2 = dmin * m2;

        const uint8_t *bq = qs + j/2;
        for (int l = 0; l < 32; l++)
            out[j + l]      = d1 * (bq[l] & 0xF) - ml1;
        for (int l = 0; l < 32; l++)
            out[j + 32 + l] = d2 * (bq[l] >> 4) - ml2;
        is += 2;
    }
}

// Initialize GPU output projection
// If GPU_QUANTIZED=1 is set, uses Q4_K weight directly on GPU (lower VRAM)
bool gpu_output_init(const uint8_t *weight_q, int D, int V, int weight_type) {
    if (g_initialized) return true;
    
    if (weight_type != GGML_TYPE_Q4_K) {
        fprintf(stderr, "GPU output proj: expected Q4_K, got %d\n", weight_type);
        return false;
    }

    g_quantized_mode = (getenv("GPU_QUANTIZED") != NULL) ? 1 : 0;

    cudaError_t ce;
    cublasStatus_t cs;

    // Create CUDA stream and cuBLAS handle
    ce = cudaStreamCreate(&g_stream);
    if (ce != cudaSuccess) { fprintf(stderr, "cudaStreamCreate: %s\n", cudaGetErrorString(ce)); return false; }
    cs = cublasCreate(&g_cublas);
    if (cs != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasCreate failed\n"); return false; }
    cs = cublasSetStream(g_cublas, g_stream);
    if (cs != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasSetStream failed\n"); return false; }

    int batch_size = 1;
    if (getenv("GPU_BATCH")) batch_size = atoi(getenv("GPU_BATCH"));
    if (batch_size < 1) batch_size = 1;
    if (batch_size > 64) batch_size = 64;
    g_max_batch = batch_size;

    if (g_quantized_mode) {
        // Quantized mode: keep Q4_K on GPU
        int64_t blocks_per_col = (D + Q4K_ELEMS_PER_BLOCK - 1) / Q4K_ELEMS_PER_BLOCK;
        size_t weight_bytes = (size_t)V * blocks_per_col * Q4K_BLOCK_SIZE;
        
        ce = cudaMalloc(&g_d_weight_q, weight_bytes);
        if (ce != cudaSuccess) {
            fprintf(stderr, "GPU output proj: cudaMalloc Q4_K weight failed (%s), falling back to CPU\n",
                    cudaGetErrorString(ce));
            g_quantized_mode = 0;
            // Fall through to try F32 mode
        } else {
            ce = cudaMemcpy(g_d_weight_q, weight_q, weight_bytes, cudaMemcpyHostToDevice);
            if (ce != cudaSuccess) {
                fprintf(stderr, "GPU output proj: cudaMemcpy Q4_K weight failed\n");
                cudaFree(g_d_weight_q); g_d_weight_q = NULL;
                g_quantized_mode = 0;
            }
        }
    }

    if (!g_quantized_mode) {
        // F32 mode: dequant Q4_K → F32 on CPU, upload to GPU
        const int QK_K = 256;
        // Q4_K weight stored in GGUF blob as [V][ceil(D/256)*144 bytes]
        // Dequant to [V][D] row-major, then cuBLAS treats as column-major [D][V] with ld=D
        float *weight_f32 = (float*)malloc((size_t)V * D * sizeof(float));
        if (!weight_f32) { fprintf(stderr, "Failed to allocate F32 weight (%zu bytes)\\n", (size_t)V * D * 4); return false; }
        
        int blocks_per_row = (D + QK_K - 1) / QK_K;  // = ceil(2048/256) = 8
        for (int v = 0; v < V; v++) {
            const uint8_t *vocab_start = weight_q + v * blocks_per_row * Q4K_BLOCK_SIZE;
            for (int b = 0; b < blocks_per_row; b++) {
                const uint8_t *block = vocab_start + b * Q4K_BLOCK_SIZE;
                float block_out[QK_K];
                dequant_block_q4_K_f32(block, block_out);
                int n_vals = (b == blocks_per_row - 1 && (D - b * QK_K) < QK_K) ? (D - b * QK_K) : QK_K;
                memcpy(weight_f32 + (size_t)v * D + b * QK_K, block_out, n_vals * sizeof(float));
            }
        }

        ce = cudaMalloc(&g_d_weight, (size_t)D * V * sizeof(float));
        if (ce != cudaSuccess) { free(weight_f32); fprintf(stderr, "cudaMalloc weight failed (%s)\n", cudaGetErrorString(ce)); return false; }
        ce = cudaMemcpy(g_d_weight, weight_f32, (size_t)D * V * sizeof(float), cudaMemcpyHostToDevice);
        free(weight_f32);
        if (ce != cudaSuccess) { fprintf(stderr, "cudaMemcpy weight: %s\n", cudaGetErrorString(ce)); return false; }
    }

    // I/O buffers (needed in both modes)
    ce = cudaMalloc(&g_d_x, (size_t)D * g_max_batch * sizeof(float));
    if (ce != cudaSuccess) { fprintf(stderr, "cudaMalloc x failed\n"); return false; }
    ce = cudaMalloc(&g_d_y, (size_t)V * g_max_batch * sizeof(float));
    if (ce != cudaSuccess) { fprintf(stderr, "cudaMalloc y failed\n"); return false; }

    g_vocab_size = V;
    g_initialized = 1;
    fprintf(stderr, "GPU output proj: initialized (%d×%d, %s mode)\n",
            D, V, g_quantized_mode ? "Q4_K quantized" : "F32 SGEMM");
    return true;
}

// Batched output projection for prefill
bool gpu_output_project_batch(const float *input, float *output, int T) {
    if (!g_initialized) { fprintf(stderr, "GPU not initialized\n"); return false; }
    if (T > g_max_batch) {
        fprintf(stderr, "GPU batch: T=%d > max=%d, truncating\n", T, g_max_batch);
        T = g_max_batch;
    }
    if (T < 1) return false;

    int D = D_MODEL;
    int V = g_vocab_size;

    cudaError_t ce = cudaMemcpyAsync(g_d_x, input, (size_t)D * T * sizeof(float),
                                      cudaMemcpyHostToDevice, g_stream);
    if (ce != cudaSuccess) return false;

    if (g_quantized_mode) {
        // Quantized mode: custom kernel for each token
        int64_t blocks_per_col = (D + Q4K_ELEMS_PER_BLOCK - 1) / Q4K_ELEMS_PER_BLOCK;
        dim3 block_dim(256);
        dim3 grid_dim((V + 255) / 256);
        
        for (int t = 0; t < T; t++) {
            quantized_output_proj_kernel<<<grid_dim, block_dim, 0, g_stream>>>(
                g_d_x + (size_t)t * D,
                g_d_weight_q,
                g_d_y + (size_t)t * V,
                D, V, (int)blocks_per_col);
        }
    } else {
        // F32 mode: batched cuBLAS SGEMM
        // weight stored as [V][D] row-major = column-major [D][V] with ld=D
        // y[V,T] = weight^T[V,D] * x[D,T]
        float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t cs = cublasSgemm(g_cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            V, T, D,
            &alpha,
            g_d_weight, D,
            g_d_x, D,
            &beta,
            g_d_y, V);
        if (cs != CUBLAS_STATUS_SUCCESS) return false;
    }

    ce = cudaMemcpyAsync(output, g_d_y, (size_t)V * T * sizeof(float),
                          cudaMemcpyDeviceToHost, g_stream);
    if (ce != cudaSuccess) return false;

    cudaStreamSynchronize(g_stream);
    return true;
}

// Single-token output projection
bool gpu_output_project(const float *input, float *output) {
    if (!g_initialized) { fprintf(stderr, "GPU not initialized\n"); return false; }
    
    int D = D_MODEL;
    int V = g_vocab_size;
    
    cudaError_t ce;
    
    ce = cudaMemcpyAsync(g_d_x, input, D * sizeof(float), cudaMemcpyHostToDevice, g_stream);
    if (ce != cudaSuccess) { fprintf(stderr, "cudaMemcpyAsync input: %s\n", cudaGetErrorString(ce)); return false; }
    
    if (g_quantized_mode) {
        int64_t blocks_per_col = (D + Q4K_ELEMS_PER_BLOCK - 1) / Q4K_ELEMS_PER_BLOCK;
        dim3 block_dim(256);
        dim3 grid_dim((V + 255) / 256);
        
        quantized_output_proj_kernel<<<grid_dim, block_dim, 0, g_stream>>>(
            g_d_x, g_d_weight_q, g_d_y, D, V, (int)blocks_per_col);
    } else {
        float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t cs = cublasSgemm(g_cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            V, 1, D,
            &alpha,
            g_d_weight, D,
            g_d_x, D,
            &beta,
            g_d_y, V);
        if (cs != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasSgemm failed\n"); return false; }
    }
    
    ce = cudaMemcpyAsync(output, g_d_y, V * sizeof(float), cudaMemcpyDeviceToHost, g_stream);
    if (ce != cudaSuccess) { fprintf(stderr, "cudaMemcpyAsync output: %s\n", cudaGetErrorString(ce)); return false; }
    
    cudaStreamSynchronize(g_stream);

    if (getenv("VERBOSE_OUTPUT_PROJ")) {
        float gpu_first[10];
        memcpy(gpu_first, output, 10 * sizeof(float));
        fprintf(stderr, "GPU logits[0..9]:");
        for (int i = 0; i < 10; i++) fprintf(stderr, " %.6f", gpu_first[i]);
        fprintf(stderr, "\n");
    }

    return true;
}

// GPU resources for GQA attention
static float *g_d_gqa_q = NULL;      // [GQA_Q_HEADS * GQA_HEAD_DIM]
static float *g_d_gqa_k = NULL;      // [GQA_TILE_SIZE * GQA_KV_HEADS * GQA_HEAD_DIM]
static float *g_d_gqa_v = NULL;      // [GQA_TILE_SIZE * GQA_KV_HEADS * GQA_HEAD_DIM]
static float *g_d_gqa_scores = NULL; // [GQA_Q_HEADS * GQA_TILE_SIZE] (per-tile buffer, reused)
static float *g_d_gqa_out = NULL;    // [GQA_Q_HEADS * GQA_HEAD_DIM]
static float *g_d_gqa_max = NULL;    // [GQA_Q_HEADS] per-head max scores
static float *g_d_gqa_sumexp = NULL; // [GQA_Q_HEADS] per-head softmax denominators
static int g_gqa_initialized = 0;

// Cleanup
void gpu_output_cleanup() {
    if (g_d_weight) cudaFree(g_d_weight);
    if (g_d_weight_q) cudaFree(g_d_weight_q);
    if (g_d_x) cudaFree(g_d_x);
    if (g_d_y) cudaFree(g_d_y);
    if (g_cublas) cublasDestroy(g_cublas);
    if (g_stream) cudaStreamDestroy(g_stream);
    // GPU GQA resources (if allocated)
    if (g_d_gqa_q) cudaFree(g_d_gqa_q);
    if (g_d_gqa_k) cudaFree(g_d_gqa_k);
    if (g_d_gqa_v) cudaFree(g_d_gqa_v);
    if (g_d_gqa_scores) cudaFree(g_d_gqa_scores);
    if (g_d_gqa_out) cudaFree(g_d_gqa_out);
    if (g_d_gqa_max) cudaFree(g_d_gqa_max);
    if (g_d_gqa_sumexp) cudaFree(g_d_gqa_sumexp);
    g_initialized = 0;
    g_gqa_initialized = 0;
    fprintf(stderr, "GPU output proj: cleaned up\n");
}

// ================================================================
// GPU GQA Attention — stream KV tiles through GPU
//
// Processes one GQA layer's attention on GPU.
// Q: [GQA_Q_HEADS, GQA_HEAD_DIM] F32 (CPU)
// K_cache: [cache_len, GQA_KV_HEADS, GQA_HEAD_DIM] F16 (CPU)
// V_cache: same layout F16
// output: [GQA_Q_HEADS, GQA_HEAD_DIM] F32 (CPU)
//
// Uses tile-streaming: processes cache_len positions in batches
// of GQA_TILE_SIZE, accumulating attention on GPU.
// ================================================================

#define GQA_TILE_SIZE 4096

// CUDA kernel: compute Q·K scores for one tile
// Each thread block: 1 Q head, 1 KV position
// Block dim: 256 (processes 256-dim head dot product)
__global__ void gqa_qk_kernel(const float *q,          // [Q_HEADS, HEAD_DIM]
                               const uint16_t *k_cache, // [TILE_SIZE, KV_HEADS, HEAD_DIM] F16
                               float *scores,           // [Q_HEADS, TILE_SIZE] output
                               int tile_start, int tile_end,
                               int kv_stride) {         // stride in F16 elements per position
    int h_q = blockIdx.y;          // which Q head
    int t_k = blockIdx.x;          // position within tile
    if (h_q >= GQA_Q_HEADS || t_k >= (tile_end - tile_start)) return;
    
    int h_kv = h_q / (GQA_Q_HEADS / GQA_KV_HEADS);  // GQA mapping
    int global_pos = tile_start + t_k;
    
    // Load Q vector (256-dim)
    __shared__ float q_shared[GQA_HEAD_DIM];
    for (int i = threadIdx.x; i < GQA_HEAD_DIM; i += blockDim.x) {
        q_shared[i] = q[h_q * GQA_HEAD_DIM + i];
    }
    __syncthreads();
    
    // Load K vector (F16 -> F32)
    float sum = 0.0f;
    int k_offset = global_pos * kv_stride + h_kv * GQA_HEAD_DIM;
    for (int i = threadIdx.x; i < GQA_HEAD_DIM; i += blockDim.x) {
        uint16_t k_f16 = k_cache[k_offset + i];
        // F16 to F32
        int sign = (k_f16 >> 15) & 1;
        int exp  = (k_f16 >> 10) & 0x1F;
        int mant = k_f16 & 0x03FF;
        float k_f32;
        if (exp == 0) {
            k_f32 = ldexpf((float)mant / 1024.0f, -14) * (sign ? -1.0f : 1.0f);
        } else if (exp == 31) {
            k_f32 = sign ? -INFINITY : INFINITY;
        } else {
            k_f32 = ldexpf(1.0f + (float)mant / 1024.0f, exp - 15) * (sign ? -1.0f : 1.0f);
        }
        sum += q_shared[i] * k_f32;
    }
    
    // Warp reduce sum
    __shared__ float red_buf[WARP_SIZE];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (lane < WARP_SIZE) {
        float w = sum;
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
            w += __shfl_xor_sync(0xFFFFFFFF, w, offset);
        if (lane == 0) red_buf[warp_id] = w;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float final = 0.0f;
        for (int i = 0; i < blockDim.x / WARP_SIZE; i++) final += red_buf[i];
        scores[h_q * (gridDim.x) + t_k] = final;  // store as raw score
    }
}

// CUDA kernel: find max score and compute softmax for all Q heads
// Then compute V weighted sum for each tile
// scores: [Q_HEADS, total_positions] raw scores (input/output — replaced with softmax probs)
// v_cache: [TILE_SIZE, KV_HEADS, HEAD_DIM] F16
// output: [Q_HEADS, HEAD_DIM] accumulated result (atomic add)
// max_scores: [Q_HEADS] per-head max (pre-computed)
// sum_exps: [Q_HEADS] per-head softmax denominator (pre-computed)
__global__ void gqa_softmax_v_kernel(const float *scores,
                                      const uint16_t *v_cache,
                                      float *output,
                                      int total_positions,
                                      float *max_scores,
                                      float *sum_exps,
                                      int tile_start, int tile_end,
                                      int kv_stride) {
    int h_q = blockIdx.y;
    int t_k = blockIdx.x;
    if (h_q >= GQA_Q_HEADS || t_k >= (tile_end - tile_start)) return;
    
    int h_kv = h_q / (GQA_Q_HEADS / GQA_KV_HEADS);
    int global_pos = tile_start + t_k;
    
    // Load raw score
    float raw = scores[(size_t)h_q * total_positions + global_pos];
    
    // Compute softmax probability
    float prob = expf(raw - max_scores[h_q]) / sum_exps[h_q];
    
    // Load V for this position (F16 → F32)
    float v_contrib[GQA_HEAD_DIM];
    int v_offset = global_pos * kv_stride + h_kv * GQA_HEAD_DIM;
    for (int i = threadIdx.x; i < GQA_HEAD_DIM; i += blockDim.x) {
        uint16_t v_f16 = v_cache[v_offset + i];
        int sign = (v_f16 >> 15) & 1;
        int exp  = (v_f16 >> 10) & 0x1F;
        int mant = v_f16 & 0x03FF;
        float v_f32;
        if (exp == 0) v_f32 = ldexpf((float)mant / 1024.0f, -14) * (sign ? -1.0f : 1.0f);
        else if (exp == 31) v_f32 = sign ? -INFINITY : INFINITY;
        else v_f32 = ldexpf(1.0f + (float)mant / 1024.0f, exp - 15) * (sign ? -1.0f : 1.0f);
        v_contrib[i] = prob * v_f32;
    }
    __syncthreads();
    
    // Accumulate into output (block-level reduction per head)
    // Each thread handles a subset of HEAD_DIM elements
    for (int i = threadIdx.x; i < GQA_HEAD_DIM; i += blockDim.x) {
        atomicAdd(&output[(size_t)h_q * GQA_HEAD_DIM + i], v_contrib[i]);
    }
}

// Reduction kernel: find max score per Q head
__global__ void gqa_find_max_kernel(const float *scores,
                                     float *max_scores,
                                     int total_positions) {
    int h_q = blockIdx.x;
    if (h_q >= GQA_Q_HEADS) return;
    
    float local_max = -1e30f;
    for (int t = threadIdx.x; t < total_positions; t += blockDim.x) {
        float s = scores[(size_t)h_q * total_positions + t];
        if (s > local_max) local_max = s;
    }
    
    // Block-level reduction
    __shared__ float red[256];
    red[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            if (red[threadIdx.x + s] > red[threadIdx.x])
                red[threadIdx.x] = red[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) max_scores[h_q] = red[0];
}

// Reduction kernel: compute softmax denominator (sum of exp(x - max))
__global__ void gqa_sumexp_kernel(const float *scores,
                                   float *sum_exps,
                                   const float *max_scores,
                                   int total_positions) {
    int h_q = blockIdx.x;
    if (h_q >= GQA_Q_HEADS) return;
    
    float local_sum = 0.0f;
    float max_s = max_scores[h_q];
    for (int t = threadIdx.x; t < total_positions; t += blockDim.x) {
        local_sum += expf(scores[(size_t)h_q * total_positions + t] - max_s);
    }
    
    __shared__ float red[256];
    red[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s /= 2) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) sum_exps[h_q] = red[0];
}

// Host-side: GPU GQA attention for one GQA layer
// Processes the full KV cache in tiles
bool gpu_gqa_attention(const float *q,            // [GQA_Q_HEADS, GQA_HEAD_DIM] F32
                       const void *k_cache,       // [cache_len, ...] F16 KV cache
                       const void *v_cache,       // same
                       int cache_len,
                       float *output) {            // [Q_HEADS, HEAD_DIM] F32 result
    if (cache_len <= 0) return false;
    
    if (!g_gqa_initialized) {
        // Allocate GPU buffers
        cudaError_t ce;
        ce = cudaMalloc(&g_d_gqa_q, (size_t)GQA_Q_HEADS * GQA_HEAD_DIM * sizeof(float));
        if (ce != cudaSuccess) return false;
        ce = cudaMalloc(&g_d_gqa_k, (size_t)GQA_TILE_SIZE * GQA_KV_HEADS * GQA_HEAD_DIM * sizeof(uint16_t));
        if (ce != cudaSuccess) { cudaFree(g_d_gqa_q); g_d_gqa_q = NULL; return false; }
        ce = cudaMalloc(&g_d_gqa_v, (size_t)GQA_TILE_SIZE * GQA_KV_HEADS * GQA_HEAD_DIM * sizeof(uint16_t));
        if (ce != cudaSuccess) { cudaFree(g_d_gqa_q); cudaFree(g_d_gqa_k); g_d_gqa_q = NULL; g_d_gqa_k = NULL; return false; }
        ce = cudaMalloc(&g_d_gqa_scores, (size_t)GQA_Q_HEADS * GQA_TILE_SIZE * sizeof(float));
        if (ce != cudaSuccess) { cudaFree(g_d_gqa_q); cudaFree(g_d_gqa_k); cudaFree(g_d_gqa_v); return false; }
        ce = cudaMalloc(&g_d_gqa_out, (size_t)GQA_Q_HEADS * GQA_HEAD_DIM * sizeof(float));
        if (ce != cudaSuccess) { cudaFree(g_d_gqa_q); cudaFree(g_d_gqa_k); cudaFree(g_d_gqa_v); cudaFree(g_d_gqa_scores); return false; }
        ce = cudaMalloc(&g_d_gqa_max, (size_t)GQA_Q_HEADS * sizeof(float));
        if (ce != cudaSuccess) { cudaFree(g_d_gqa_q); cudaFree(g_d_gqa_k); cudaFree(g_d_gqa_v); cudaFree(g_d_gqa_scores); cudaFree(g_d_gqa_out); return false; }
        ce = cudaMalloc(&g_d_gqa_sumexp, (size_t)GQA_Q_HEADS * sizeof(float));
        if (ce != cudaSuccess) { cudaFree(g_d_gqa_q); cudaFree(g_d_gqa_k); cudaFree(g_d_gqa_v); cudaFree(g_d_gqa_scores); cudaFree(g_d_gqa_out); cudaFree(g_d_gqa_max); return false; }
        g_gqa_initialized = 1;
        fprintf(stderr, "GPU GQA: initialized (tile=%d, %.1f MB total)\n",
                GQA_TILE_SIZE,
                (double)((size_t)GQA_Q_HEADS * GQA_HEAD_DIM * sizeof(float) * 2 +
                         (size_t)GQA_TILE_SIZE * GQA_KV_HEADS * GQA_HEAD_DIM * sizeof(uint16_t) * 2 +
                         (size_t)GQA_Q_HEADS * GQA_TILE_SIZE * sizeof(float)) / 1048576.0);
    }
    
    int kv_stride = GQA_KV_HEADS * GQA_HEAD_DIM;  // elements per position in F16 cache
    
    // Upload Q
    cudaMemcpyAsync(g_d_gqa_q, q, (size_t)GQA_Q_HEADS * GQA_HEAD_DIM * sizeof(float),
                    cudaMemcpyHostToDevice, g_stream);
    
    // Allocate global score buffer for all positions
    static float *g_all_scores = NULL;
    static size_t g_all_scores_size = 0;
    size_t needed = (size_t)GQA_Q_HEADS * cache_len * sizeof(float);
    if (needed > g_all_scores_size) {
        if (g_all_scores) cudaFree(g_all_scores);
        cudaMalloc(&g_all_scores, needed);
        g_all_scores_size = needed;
    }
    
    // Pass 1: Compute Q·K scores tile by tile, store in global buffer
    {
        int n_tiles = (cache_len + GQA_TILE_SIZE - 1) / GQA_TILE_SIZE;
        for (int tile = 0; tile < n_tiles; tile++) {
            int tile_start = tile * GQA_TILE_SIZE;
            int tile_end = (tile_start + GQA_TILE_SIZE < cache_len) ? (tile_start + GQA_TILE_SIZE) : cache_len;
            int tile_size = tile_end - tile_start;
            
            // Upload K tile
            size_t k_bytes = (size_t)tile_size * kv_stride * sizeof(uint16_t);
            const uint16_t *k_src = (const uint16_t *)k_cache + (size_t)tile_start * kv_stride;
            cudaMemcpyAsync(g_d_gqa_k, k_src, k_bytes, cudaMemcpyHostToDevice, g_stream);
            
            // Launch Q·K kernel — store into global buffer at correct offset
            float *score_offset = g_all_scores + (size_t)GQA_Q_HEADS * tile_start;
            dim3 block_dim(256);
            dim3 grid_dim(tile_size, GQA_Q_HEADS);
            gqa_qk_kernel<<<grid_dim, block_dim, 0, g_stream>>>(
                g_d_gqa_q, (const uint16_t *)g_d_gqa_k, score_offset,
                tile_start, tile_end, kv_stride);
        }
    }
    
    // Step 2: Find max score per Q head (GPU reduction)
    gqa_find_max_kernel<<<GQA_Q_HEADS, 256, 0, g_stream>>>(
        g_all_scores, g_d_gqa_max, cache_len);
    
    // Step 3: Compute softmax denominator per Q head (GPU reduction)
    gqa_sumexp_kernel<<<GQA_Q_HEADS, 256, 0, g_stream>>>(
        g_all_scores, g_d_gqa_sumexp, g_d_gqa_max, cache_len);
    
    cudaStreamSynchronize(g_stream);
    
    // Step 4: Compute V weighted sum on GPU (tile by tile)
    cudaMemsetAsync(g_d_gqa_out, 0, (size_t)GQA_Q_HEADS * GQA_HEAD_DIM * sizeof(float), g_stream);
    
    {
        int n_tiles = (cache_len + GQA_TILE_SIZE - 1) / GQA_TILE_SIZE;
        for (int tile = 0; tile < n_tiles; tile++) {
            int tile_start = tile * GQA_TILE_SIZE;
            int tile_end = (tile_start + GQA_TILE_SIZE < cache_len) ? (tile_start + GQA_TILE_SIZE) : cache_len;
            int tile_size = tile_end - tile_start;
            
            // Upload V tile
            size_t v_bytes = (size_t)tile_size * kv_stride * sizeof(uint16_t);
            const uint16_t *v_src = (const uint16_t *)v_cache + (size_t)tile_start * kv_stride;
            cudaMemcpyAsync(g_d_gqa_v, v_src, v_bytes, cudaMemcpyHostToDevice, g_stream);
            
            // Launch softmax+V kernel
            dim3 block_dim(256);
            dim3 grid_dim(tile_size, GQA_Q_HEADS);
            gqa_softmax_v_kernel<<<grid_dim, block_dim, 0, g_stream>>>(
                g_all_scores, (const uint16_t *)g_d_gqa_v, g_d_gqa_out,
                cache_len, g_d_gqa_max, g_d_gqa_sumexp,
                tile_start, tile_end, kv_stride);
        }
    }
    
    // Download final output
    cudaMemcpyAsync(output, g_d_gqa_out,
                    (size_t)GQA_Q_HEADS * GQA_HEAD_DIM * sizeof(float),
                    cudaMemcpyDeviceToHost, g_stream);
    
    cudaStreamSynchronize(g_stream);
    return true;
}
