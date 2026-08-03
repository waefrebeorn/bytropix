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

/* y[M,N] = x[M,K] @ w[K,N]  (row-major F32). Returns 1 on success
 * (GPU used), 0 if the GPU path is unavailable (caller falls back). */
int gpu_barun_matmul(float *y, const float *w, const float *x,
                     int M, int N, int K);

#ifdef __cplusplus
}
#endif
#endif
