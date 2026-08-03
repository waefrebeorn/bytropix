/*
 * gpu_barun.h -- the GPU backend for BarunLM training (C linkage).
 * The trainer dispatches through this: CPU fallback when no GPU,
 * cuBLAS SGEMM when CUDA is present.
 */
#ifndef GPU_BARUN_H
#define GPU_BARUN_H

#ifdef __cplusplus
extern "C" {
#endif

int gpu_barun_init(void);
void gpu_barun_free(void);
int gpu_barun_ready(void);
/* call after the optimizer updates the weights: the GPU weight cache
 * re-uploads on the next matmul */
void gpu_barun_mark_weights_dirty(void);

/* y[M,N] = x[M,K] @ w[K,N]  (row-major F32). Returns 1 on success
 * (GPU used), 0 if the GPU path is unavailable (caller falls back). */
int gpu_barun_matmul(float *y, const float *w, const float *x,
                     int M, int N, int K);

/* y[M,N] = a[K,M]^T @ b[K,N]  (row-major F32) -- the backward's
 * weight-gradient outer products. Returns 1 on success, 0 otherwise. */
int gpu_barun_matmul_tx(float *y, const float *a, const float *b,
                        int M, int N, int K);

/* y[M,N] = x[M,K] @ w[K,N] with w STORED [K,N] (no transpose) -- the
 * backward's input-gradient products. Returns 1 on success, 0 otherwise. */
int gpu_barun_matmul_nt(float *y, const float *w, const float *x,
                        int M, int N, int K);

/* The Muon Newton-Schulz 5 orthogonalization: X[rows,cols] in-place,
 * tall matrices transposed, Frobenius-renormalized per iteration.
 * Returns 1 on success, 0 otherwise (caller falls back to CPU). */
int gpu_barun_ns5(float *X, int rows, int cols);

/* The Gram Newton-Schulz (Zhang/Amsel/Chen/Dao 2026): the square-space
 * iteration -- one rectangular GEMM at each end, square GEMMs inside.
 * Mathematically identical to the standard NS; ~5x fewer rectangular
 * FLOPs. Returns 1 on success, 0 otherwise. */
int gpu_barun_ns5_gram(float *X, int rows, int cols);

/* The hybrid GQA attention: q [seq, heads*dim] (the head h's slice at
 * the column offset h*dim), the single shared k/v [seq, dim]. The
 * causal + local-window mask, the 1/sqrt(dim) scale, the softmax.
 * Matches the bp's CPU reference to 1e-3 (the FD oracle). */
int gpu_barun_attn(float *out, const float *q, const float *k, const float *v,
                   int seq, int heads, int dim, int local_win, int is_full);

#ifdef __cplusplus
}
#endif
#endif
