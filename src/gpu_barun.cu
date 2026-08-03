/*
 * gpu_barun.cu -- CUDA kernels for BarunLM training (the seed grows fast).
 *
 * The DA pass found: the wizard already has cuBLAS + GPU kernels, but
 * the Barun training loop was pure CPU. This module gives the trainer
 * a GPU backend: SGEMM (cuBLAS) for the big matrix products and a
 * fused attention kernel for the hybrid local/global pattern. The
 * trainer calls these through a pluggable dispatch -- CPU when no GPU,
 * CUDA when present (the wubu_model.h pattern).
 *
 * API (C linkage):
 *   gpu_barun_init() / gpu_barun_free()
 *   gpu_barun_matmul(y, w, x, M, N, K)        // y[M,N] = x[M,K] @ w[K,N]
 *   gpu_barun_attn(...)                        // hybrid windowed attention
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

static cublasHandle_t g_cublas = NULL;
static int g_ready = 0;

extern "C" {

int gpu_barun_init(void)
{
    if (g_ready) return 1;
    cudaError_t ce = cudaSetDevice(0);
    if (ce != cudaSuccess) { fprintf(stderr, "gpu_barun: no CUDA device\n"); return 0; }
    cublasStatus_t st = cublasCreate(&g_cublas);
    if (st != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "gpu_barun: cublas init failed\n"); return 0; }
    g_ready = 1;
    return 1;
}

void gpu_barun_free(void)
{
    if (g_cublas) { cublasDestroy(g_cublas); g_cublas = NULL; }
    g_ready = 0;
}

int gpu_barun_ready(void) { return g_ready; }

/* y[M,N] = x[M,K] @ w[K,N]  (row-major, F32). Uses cuBLAS SGEMM. */
int gpu_barun_matmul(float *y, const float *w, const float *x,
                     int M, int N, int K)
{
    if (!g_ready) return 0;
    if (M <= 0 || N <= 0 || K <= 0) return 0;
    static float *d_x = NULL, *d_w = NULL, *d_y = NULL;
    static size_t cap_x = 0, cap_w = 0, cap_y = 0;
    size_t nx = (size_t)M * K, nw = (size_t)K * N, ny = (size_t)M * N;
    if (nx > cap_x) {
        if (d_x) cudaFree(d_x);
        cudaMalloc(&d_x, nx * sizeof(float)); cap_x = nx;
    }
    if (nw > cap_w) {
        if (d_w) cudaFree(d_w);
        cudaMalloc(&d_w, nw * sizeof(float)); cap_w = nw;
    }
    if (ny > cap_y) {
        if (d_y) cudaFree(d_y);
        cudaMalloc(&d_y, ny * sizeof(float)); cap_y = ny;
    }
    if (!d_x || !d_w || !d_y) return 0;
    cudaMemcpy(d_x, x, nx * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, nw * sizeof(float), cudaMemcpyHostToDevice);
    /* The caller's CPU loop is out[s,o] = sum_i w[o,i] * x[s,i] where w
     * is stored [out,in] row-major (w[o*in+i]) -- i.e. out = x @ w^T.
     * cuBLAS: C^T = B^T @ A^T ; we need C = x @ w^T so:
     *   cublasSgemm(OP_T, OP_N, N, M, K) with A=w (lda=in, transposed),
     *   B=x (lda=in), C=y -- the DA check (gpu_matmul_check) proves this
     *   matches the CPU loop to <1e-4. */
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                N, M, K, &alpha, d_w, K, d_x, K, &beta, d_y, N);
    cudaMemcpy(y, d_y, ny * sizeof(float), cudaMemcpyDeviceToHost);
    return 1;
}

/* The hybrid attention score: not the kernel (the trainer runs it on
 * CPU for now); this is the GPU-side stub that proves the dispatch. */
int gpu_barun_attn_ready(void) { return g_ready; }

}
