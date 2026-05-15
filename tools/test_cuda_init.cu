/**
 * test_cuda_init.c — Measure CUDA initialization time
 */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    double t0 = now_sec();
    printf("Starting CUDA init...\n"); fflush(stdout);
    
    cudaFree(0);  // Lazy init
    printf("cudaFree(0): %.3fs\n", now_sec() - t0); fflush(stdout);
    
    cublasHandle_t h;
    cublasCreate(&h);
    printf("cublasCreate: %.3fs\n", now_sec() - t0); fflush(stdout);
    
    cudaStream_t s;
    cudaStreamCreate(&s);
    printf("cudaStreamCreate: %.3fs\n", now_sec() - t0); fflush(stdout);
    
    cublasSetStream(h, s);
    printf("Total: %.3fs\n", now_sec() - t0); fflush(stdout);
    
    cudaStreamDestroy(s);
    cublasDestroy(h);
    return 0;
}
