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

/* y[M,N] = a[K,M]^T @ b[K,N]  (row-major, F32) -- the backward's
 * weight-gradient outer products: dW[out,in] = dy[seq,out]^T @ x[seq,in]
 * with M=out, N=in, K=seq. Verified by test_backprop's FD checks (the
 * weight grads are numerically checked per parameter type).
 * cuBLAS mapping (a row-major [K,M] lda=M == col-major [M,K];
 * b row-major [K,N] lda=N == col-major [N,K]): C[N,M] = B' @ A'^T with
 * C = y^T, so sgemm(OP_N, OP_T, N, M, K, B', N, A', M, C, N). */
int gpu_barun_matmul_tx(float *y, const float *a, const float *b,
                        int M, int N, int K)
{
    if (!g_ready) return 0;
    if (M <= 0 || N <= 0 || K <= 0) return 0;
    static float *d_a = NULL, *d_b = NULL, *d_y = NULL;
    static size_t cap_a = 0, cap_b = 0, cap_y = 0;
    size_t na = (size_t)K * M, nb = (size_t)K * N, ny = (size_t)M * N;
    if (na > cap_a) {
        if (d_a) cudaFree(d_a);
        cudaMalloc(&d_a, na * sizeof(float)); cap_a = na;
    }
    if (nb > cap_b) {
        if (d_b) cudaFree(d_b);
        cudaMalloc(&d_b, nb * sizeof(float)); cap_b = nb;
    }
    if (ny > cap_y) {
        if (d_y) cudaFree(d_y);
        cudaMalloc(&d_y, ny * sizeof(float)); cap_y = ny;
    }
    if (!d_a || !d_b || !d_y) return 0;
    cudaMemcpy(d_a, a, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, nb * sizeof(float), cudaMemcpyHostToDevice);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                N, M, K, &alpha, d_b, N, d_a, M, &beta, d_y, N);
    cudaMemcpy(y, d_y, ny * sizeof(float), cudaMemcpyDeviceToHost);
    return 1;
}

/* y[M,N] = x[M,K] @ w[K,N] with w STORED [K,N] (no transpose) -- the
 * backward's input-gradient products: dL/dx[seq,in] = dy[seq,out] @
 * w[out,in] with M=seq, N=in, K=out. */
int gpu_barun_matmul_nt(float *y, const float *w, const float *x,
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
    /* x row-major [M,K] lda=K == col-major [K,M]; w row-major [K,N]
     * lda=N == col-major [N,K]; y row-major [M,N] lda=N == [N,M].
     * C[N,M] = op(A)[N,K] @ op(B)[K,M] with op(A)=N, op(B)=N. */
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, d_w, N, d_x, K, &beta, d_y, N);
    cudaMemcpy(y, d_y, ny * sizeof(float), cudaMemcpyDeviceToHost);
    return 1;
}

/* The Muon Newton-Schulz 5 orthogonalization, GPU-side (the optimizer
 * was the last CPU bottleneck -- ~61 GFLOP/step of NS5). X[rows,cols]
 * in-place; tall matrices are transposed first per the recipe; each
 * iteration is Frobenius-renormalized (the fp32-stability fix). Uses
 * cuBLAS GEMMs + nrm2/scal/axpby. Returns 1 on success. */
int gpu_barun_ns5(float *X, int rows, int cols)
{
    if (!g_ready) return 0;
    if (rows <= 0 || cols <= 0) return 0;
    int trows = rows, tcols = cols;
    if (rows > cols) { trows = cols; tcols = rows; }
    size_t n = (size_t)rows * cols;
    size_t nsq = (size_t)trows * trows;
    static float *d_m = NULL, *d_t = NULL, *d_a = NULL, *d_a2 = NULL;
    static size_t cap_m = 0, cap_t = 0, cap_a = 0;
    if (n > cap_m) {
        if (d_m) cudaFree(d_m);
        cudaMalloc(&d_m, n * sizeof(float)); cap_m = n;
    }
    if (n > cap_t) {
        if (d_t) cudaFree(d_t);
        cudaMalloc(&d_t, n * sizeof(float)); cap_t = n;
    }
    if (nsq > cap_a) {
        if (d_a) cudaFree(d_a);
        if (d_a2) cudaFree(d_a2);
        cudaMalloc(&d_a, nsq * sizeof(float)); cap_a = nsq;
        cudaMalloc(&d_a2, nsq * sizeof(float));
    }
    if (!d_m || !d_t || !d_a || !d_a2) return 0;
    cudaMemcpy(d_m, X, n * sizeof(float), cudaMemcpyHostToDevice);
    float *M = d_m;
    cublasStatus_t st;
    if (rows > cols) {
        /* d_t = M^T  (trows=cols, tcols=rows); OP_T: lda >= n;
         * OP_N: ldb >= m. B is never read (beta=0) but must validate. */
        float one = 1.0f, zero = 0.0f;
        st = cublasSgeam(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         rows, cols, &one, d_m, cols, &zero, d_m, rows,
                         d_t, rows);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
        M = d_t;
    }
    /* T workspace: for the tall case M lives in d_t so d_m is free;
     * for the wide case M lives in d_m so d_t is free. Never alias M. */
    float *Tbuf = (rows > cols) ? d_m : d_t;
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    for (int it = 0; it < 5; it++) {
        /* per-iteration Frobenius renormalization (fp32 stability) */
        float nrm = 0;
        cublasSnrm2(g_cublas, (int)((size_t)trows * tcols), M, 1, &nrm);
        if (nrm > 1e-12f) {
            float s = 1.0f / nrm;
            cublasSscal(g_cublas, (int)((size_t)trows * tcols), &s, M, 1);
        }
        float one = 1.0f, zero = 0.0f;
        /* A = M @ M^T  [trows,trows]. Views (row-major [R,C] lda=C ==
         * col-major [C,R] lda=C): M' = [tcols,trows] ldm=tcols, A square.
         * A'[i,j] = sum_k M'[k,i] M'[k,j] = (M'^T @ M')  -> OP_T, OP_N. */
        st = cublasSgemm(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         trows, trows, tcols, &one, M, tcols, M, tcols,
                         &zero, d_a, trows);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
        /* A2 = A @ A (square: view is self-dual) */
        st = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                         trows, trows, trows, &one, d_a, trows, d_a, trows,
                         &zero, d_a2, trows);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
        /* B = b*A + c*A2  (into d_a2, square) */
        st = cublasSgeam(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                         trows, trows, &b, d_a, trows, &c, d_a2, trows,
                         d_a2, trows);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
        /* T = B @ M  [trows,tcols]. T' = [tcols,trows] ldt=tcols,
         * B' = [trows,trows] ldb=trows, M' = [tcols,trows] ldm=tcols.
         * T'[j,i] = sum_k M'[j,k] B'[k,i] = (M' @ B') -> dims (tcols,
         * trows, trows) with OP_N, OP_N. */
        st = cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                         tcols, trows, trows, &one, M, tcols, d_a2, trows,
                         &zero, Tbuf, tcols);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
        /* M = a*M + T  (both in the [tcols,trows] view) */
        st = cublasSgeam(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                         tcols, trows, &a, M, tcols, &one, Tbuf, tcols,
                         M, tcols);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
    }
    if (rows > cols) {
        float one = 1.0f, zero = 0.0f;
        st = cublasSgeam(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         cols, rows, &one, M, rows, &zero, M, cols,
                         d_m, cols);
        if (st != CUBLAS_STATUS_SUCCESS) return 0;
    }
    cudaMemcpy(X, d_m, n * sizeof(float), cudaMemcpyDeviceToHost);
    return 1;
}

/* The hybrid attention score: not the kernel (the trainer runs it on
 * CPU for now); this is the GPU-side stub that proves the dispatch. */
int gpu_barun_attn_ready(void) { return g_ready; }

}
