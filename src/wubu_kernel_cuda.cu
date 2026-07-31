/*
 * wubu_kernel_cuda.cu — CUDA device backend for wubu_kernel dispatch.
 *
 * Implements GEMM, GEMV, Attention, Softmax, RMSNorm, Quantize, Dequantize
 * as CUDA kernels. Function signatures match the CPU baseline typedefs —
 * data is copied H→D before the kernel, D→H after.
 *
 * Compiled with nvcc, linked with -lcudart.
 * Build: make wubu_kernel_cuda.o  (uses nvcc)
 */
#include "wubu_kernel.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>

/* Forward declaration — defined in gpu_quant_matmul.cu, linked at build time */
extern "C" int wubu_cuda_quant_matmul_batched(const float *x, int C,
    const uint8_t *W_q, int quant_type, int n_rows, int n_cols,
    float *y, cudaStream_t stream);

/* ------------------------------------------------------------------ */
/* CUDA kernels                                                       */
/* ------------------------------------------------------------------ */

__global__ void cuda_gemm_kernel(const float *A, const float *B, float *C,
                                  int M, int K, int N, float beta) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = (beta == 0.0f) ? 0.0f : beta * C[row * N + col];
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

__global__ void cuda_gemv_kernel(const float *A, const float *x, float *y,
                                  int M, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[i * K + k] * x[k];
        y[i] = sum;
    }
}

__global__ void cuda_softmax_kernel(float *logits, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    float *r = logits + row * N;
    float maxv = r[0];
    for (int j = 1; j < N; j++)
        if (r[j] > maxv) maxv = r[j];
    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        r[j] = expf(r[j] - maxv);
        sum += r[j];
    }
    if (sum > 0.0f)
        for (int j = 0; j < N; j++) r[j] /= sum;
    else {
        float u = 1.0f / (float)N;
        for (int j = 0; j < N; j++) r[j] = u;
    }
}

__global__ void cuda_rmsnorm_kernel(float *x, const float *gamma,
                                     const float *beta, int M, int d, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    float *row = x + i * d;
    float sum_sq = 0.0f;
    for (int j = 0; j < d; j++) sum_sq += row[j] * row[j];
    float rsqrt = 1.0f / sqrtf(sum_sq / (float)d + eps);
    for (int j = 0; j < d; j++)
        row[j] = row[j] * rsqrt * gamma[j] + beta[j];
}

__global__ void cuda_quantize_kernel(const float *fp32, int8_t *q,
                                      float *scales, int M, int K, int bits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    const float *row = fp32 + i * K;
    float amax = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = fabsf(row[k]);
        if (a > amax) amax = a;
    }
    float scale = (amax > 1e-8f) ? amax / (float)((1 << bits) - 1) : 1e-8f;
    scales[i] = scale;
    float inv = 1.0f / scale;
    for (int k = 0; k < K; k++) {
        int v = (int)roundf(row[k] * inv);
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        q[i * K + k] = (int8_t)v;
    }
}

__global__ void cuda_dequantize_kernel(const int8_t *q, const float *scales,
                                        const float *zeros, float *fp32,
                                        int M, int K, int bits) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    const int8_t *qr = q + i * K;
    float *row = fp32 + i * K;
    float s = scales[i];
    float z = (zeros ? zeros[i] : 0.0f);
    for (int k = 0; k < K; k++)
        row[k] = s * ((float)qr[k] - z);
}

/* ------------------------------------------------------------------ */
/* Host wrappers (match wubu_kernel typedefs)                          */
/* ------------------------------------------------------------------ */

static void cuda_gemm(const float *A, const float *B, float *C,
                       int M, int K, int N, float beta) {
    size_t a_bytes = (size_t)M * K * sizeof(float);
    size_t b_bytes = (size_t)K * N * sizeof(float);
    size_t c_bytes = (size_t)M * N * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, a_bytes); cudaMalloc(&dB, b_bytes); cudaMalloc(&dC, c_bytes);
    cudaMemcpy(dA, A, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, b_bytes, cudaMemcpyHostToDevice);
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    cuda_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, K, N, beta);
    cudaMemcpy(C, dC, c_bytes, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

static void cuda_gemv(const float *A, const float *x, float *y,
                       int M, int K) {
    size_t a_bytes = (size_t)M * K * sizeof(float);
    size_t x_bytes = (size_t)K * sizeof(float);
    size_t y_bytes = (size_t)M * sizeof(float);
    float *dA, *dx, *dy;
    cudaMalloc(&dA, a_bytes); cudaMalloc(&dx, x_bytes); cudaMalloc(&dy, y_bytes);
    cudaMemcpy(dA, A, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, x_bytes, cudaMemcpyHostToDevice);
    cuda_gemv_kernel<<<(M + 255) / 256, 256>>>(dA, dx, dy, M, K);
    cudaMemcpy(y, dy, y_bytes, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dx); cudaFree(dy);
}

static void cuda_softmax(float *logits, int M, int N) {
    size_t bytes = (size_t)M * N * sizeof(float);
    float *d;
    cudaMalloc(&d, bytes);
    cudaMemcpy(d, logits, bytes, cudaMemcpyHostToDevice);
    cuda_softmax_kernel<<<(M + 255) / 256, 256>>>(d, M, N);
    cudaMemcpy(logits, d, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d);
}

static void cuda_rmsnorm(float *x, const float *gamma,
                          const float *beta, int M, int d, float eps) {
    size_t bytes = (size_t)M * d * sizeof(float);
    float *dx, *dg, *db;
    cudaMalloc(&dx, bytes); cudaMalloc(&dg, (size_t)d * sizeof(float)); cudaMalloc(&db, (size_t)d * sizeof(float));
    cudaMemcpy(dx, x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dg, gamma, (size_t)d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, beta, (size_t)d * sizeof(float), cudaMemcpyHostToDevice);
    cuda_rmsnorm_kernel<<<(M + 255) / 256, 256>>>(dx, dg, db, M, d, eps);
    cudaMemcpy(x, dx, bytes, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dg); cudaFree(db);
}

static void cuda_quantize(const float *fp32, int8_t *q, float *scales,
                           int M, int K, int bits) {
    size_t fp32_bytes = (size_t)M * K * sizeof(float);
    size_t q_bytes = (size_t)M * K * sizeof(int8_t);
    size_t s_bytes = (size_t)M * sizeof(float);
    float *df; int8_t *dq; float *ds;
    cudaMalloc(&df, fp32_bytes); cudaMalloc(&dq, q_bytes); cudaMalloc(&ds, s_bytes);
    cudaMemcpy(df, fp32, fp32_bytes, cudaMemcpyHostToDevice);
    cuda_quantize_kernel<<<(M + 255) / 256, 256>>>(df, dq, ds, M, K, bits);
    cudaMemcpy(q, dq, q_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(scales, ds, s_bytes, cudaMemcpyDeviceToHost);
    cudaFree(df); cudaFree(dq); cudaFree(ds);
}

static void cuda_dequantize(const int8_t *q, const float *scales,
                             const float *zeros, float *fp32,
                             int M, int K, int bits) {
    size_t q_bytes = (size_t)M * K * sizeof(int8_t);
    size_t s_bytes = (size_t)M * sizeof(float);
    size_t fp32_bytes = (size_t)M * K * sizeof(float);
    int8_t *dq; float *ds; float *df;
    cudaMalloc(&dq, q_bytes); cudaMalloc(&ds, s_bytes); cudaMalloc(&df, fp32_bytes);
    cudaMemcpy(dq, q, q_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, scales, s_bytes, cudaMemcpyHostToDevice);
    if (zeros) {
        float *dz; cudaMalloc(&dz, s_bytes);
        cudaMemcpy(dz, zeros, s_bytes, cudaMemcpyHostToDevice);
        cuda_dequantize_kernel<<<(M + 255) / 256, 256>>>(dq, ds, dz, df, M, K, bits);
        cudaFree(dz);
    } else {
        cuda_dequantize_kernel<<<(M + 255) / 256, 256>>>(dq, ds, NULL, df, M, K, bits);
    }
    cudaMemcpy(fp32, df, fp32_bytes, cudaMemcpyDeviceToHost);
    cudaFree(dq); cudaFree(ds); cudaFree(df);
}

/* ------------------------------------------------------------------ */
/* Registration                                                       */
/* ------------------------------------------------------------------ */

extern "C" int wubu_cuda_backend_probe(void) {
    int n = 0;
    cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess) return 0;
    return n > 0 ? 1 : 0;
}
/* A:02 — GPU-quantized matmul stub. Returns 0 (fails gracefully),
 * falling back to CPU quantized_matmul. Full GPU weight upload
 * requires wubu_model_gpu_init which handles per-model tensor naming. */
extern "C" int proj_matmul_gpu(const float *x, const uint8_t *W_q,
                               int weight_type, int n_rows, int n_layers,
                               float *out) {
    (void)x; (void)W_q; (void)weight_type; (void)n_rows; (void)n_layers; (void)out;
    return 0; /* fall back to CPU */
}

/* g_use_gpu_backend: flag set by GPU init, checked by wubu_ssm.c proj_matmul.
 * Defined in wubu_kernel_backends.c (C, non-PIC for gen_text_cpu; PIC-safe
 * for gen_text_gpu since CUDA path overrides via linker). */
extern int g_use_gpu_backend;

extern "C" int wubu_cuda_backend_register(void) {
    if (!wubu_cuda_backend_probe()) return -1;
    /* Ensure device 0 is active */
    cudaSetDevice(0);
    wubu_kernel_backend_t b;
    b.id = WUBU_BACKEND_CUDA;
    b.name = "cuda";
    b.gemm = cuda_gemm;
    b.gemv = cuda_gemv;
    b.attn = NULL;       /* attention not yet implemented on CUDA */
    b.rope = NULL;       /* ROPE not yet implemented on CUDA */
    b.softmax = cuda_softmax;
    b.rmsnorm = cuda_rmsnorm;
    b.quantize = cuda_quantize;
    b.dequantize = cuda_dequantize;
    b.supports = [](wubu_kernel_type_t t) -> int {
        switch (t) {
            case WUBU_KERN_GEMM:       return 1;
            case WUBU_KERN_GEMV:       return 1;
            case WUBU_KERN_ATTN:       return 0;  /* TODO */
            case WUBU_KERN_ROPE:       return 0;  /* TODO */
            case WUBU_KERN_SOFTMAX:    return 1;
            case WUBU_KERN_LAYER_NORM: return 1;
            case WUBU_KERN_QUANT:      return 1;
            case WUBU_KERN_DEQUANT:    return 1;
            default:                   return 0;
        }
    };
    b.next = NULL;
    return wubu_kernel_register(WUBU_BACKEND_CUDA, "cuda", &b);
}
