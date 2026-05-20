/**
 * Minimal cuBLAS test: matmul [1, 2048] @ [2048, 8192] with F32 data.
 * Same call pattern as wubu_cuda_matmul.
 */
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    cudaSetDevice(0);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);
    
    const int M = 1, K = 2048, N = 8192;
    
    // Allocate
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Fill A with 1s, B with random
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    for (int i = 0; i < M*K; i++) h_A[i] = 1.0f;
    srand(42);
    for (int i = 0; i < K*N; i++) h_B[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    
    cudaMemcpyAsync(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    
    // Same call as wubu_cuda_matmul
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t cs = cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, K,
        d_A, K,
        &beta,
        d_C, N);
    
    cudaStreamSynchronize(stream);
    printf("cuBLAS status: %d (0=success)\n", cs);
    
    // Check output
    float *h_C = (float*)malloc(M * N * sizeof(float));
    cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    printf("C[0..4]: %.4f %.4f %.4f %.4f %.4f\n", h_C[0], h_C[1], h_C[2], h_C[3], h_C[4]);
    printf("C[1000]: %.4f\n", h_C[1000]);
    
    // Also test with tensor cores disabled
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    cudaMemset(d_C, 0, M*N*sizeof(float));
    cs = cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, K,
        d_A, K,
        &beta,
        d_C, N);
    cudaStreamSynchronize(stream);
    printf("Default math mode: %d\n", cs);
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    return 0;
}
