#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Model constants
#define D_MODEL 2048
#define DT_RANK 32
#define SSM_D_STATE 128
#define CONV_DIM 8192
#define VALUE_DIM 4096
#define KEY_DIM 2048
#define CONV_KERNEL 4
#define SSM_K_HEADS 16
#define SSM_V_HEADS 32

#include "gguf_reader.h"
#include "cuda_kernels.h"

static double cos_sim_f(const float *a, const float *b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    double denom = sqrt(na) * sqrt(nb);
    return (denom > 1e-30) ? dot / denom : 1.0;
}

int main(int argc, char **argv) {
    const char *gguf_path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    int layer = argc > 2 ? atoi(argv[2]) : 0;
    
    gguf_ctx *ctx = gguf_open(gguf_path);
    if (!ctx) return 1;

    char name_beta[64];
    snprintf(name_beta, 64, "blk.%d.ssm_beta.weight", layer);
    
    gguf_tensor_info *t = gguf_find_tensor(ctx, name_beta);
    if (!t) { fprintf(stderr, "MISSING %s\n", name_beta); return 1; }
    printf("dims: %ld %ld  n_dims=%d  ggml_type=%d\n", 
           (long)t->dims[0], (long)t->dims[1], t->n_dims, t->ggml_type);
    
    int64_t n_elems = t->dims[0] * t->dims[1];
    float *w = (float*)malloc(n_elems * sizeof(float));
    gguf_read_tensor_f32(ctx, t, w, n_elems);
    gguf_close(ctx);
    
    const int dr = DT_RANK;
    float *x = (float*)malloc(D_MODEL * sizeof(float));
    srand(42);
    for (int i = 0; i < D_MODEL; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    
    // Method 1: Row-major CPU matmul W[D_MODEL, dr]
    float *cpu_row = (float*)calloc(dr, sizeof(float));
    for (int j = 0; j < dr; j++)
        for (int i = 0; i < D_MODEL; i++)
            cpu_row[j] += x[i] * w[(size_t)i * dr + j];
    
    // Method 2: Transposed CPU matmul W[dr, D_MODEL] row-major
    float *cpu_col = (float*)calloc(dr, sizeof(float));
    for (int j = 0; j < dr; j++)
        for (int i = 0; i < D_MODEL; i++)
            cpu_col[j] += x[i] * w[(size_t)j * D_MODEL + i];
    
    printf("\n=== CPU Reference ===\n");
    printf("Row-major out[0..4]: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", cpu_row[i]);
    printf("\nCol-major out[0..4]: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", cpu_col[i]);
    printf("\nCos-sim row vs col: %.8f\n", cos_sim_f(cpu_row, cpu_col, dr));
    
    // Method 3: cuBLAS
    cublasHandle_t handle;
    cudaStream_t stream;
    wubu_cuda_init(&handle, &stream);
    
    float *d_x, *d_w, *d_cublas;
    cudaMalloc(&d_x, D_MODEL * sizeof(float));
    cudaMalloc(&d_w, n_elems * sizeof(float));
    cudaMalloc(&d_cublas, dr * sizeof(float));
    cudaMemcpy(d_x, x, D_MODEL * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, n_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_cublas, 0, dr * sizeof(float));
    
    wubu_cuda_matmul(handle, d_x, 1, D_MODEL, d_w, dr, d_cublas, 1.0f, 0.0f);
    cudaStreamSynchronize(stream);
    
    float *cublas_out = (float*)malloc(dr * sizeof(float));
    cudaMemcpy(cublas_out, d_cublas, dr * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("\n=== cuBLAS ===\n");
    printf("cublas_out[0..4]: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", cublas_out[i]);
    printf("\nCos-sim cublas vs row-major: %.8f\n", cos_sim_f(cublas_out, cpu_row, dr));
    printf("Cos-sim cublas vs col-major: %.8f\n", cos_sim_f(cublas_out, cpu_col, dr));
    
    // Method 4: Fused kernel
    float *d_fused, *fused_out;
    cudaMalloc(&d_fused, 2 * dr * sizeof(float));
    fused_out = (float*)malloc(2 * dr * sizeof(float));
    
    ssm_beta_alpha_fused_decode_wrapper(stream, d_x, d_w, d_w,
        NULL, NULL, d_fused, d_fused + dr, dr);
    cudaStreamSynchronize(stream);
    cudaMemcpy(fused_out, d_fused, 2 * dr * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("\n=== Fused kernel ===\n");
    printf("fused[0..4]: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", fused_out[i]);
    printf("\nCos-sim fused vs row-major: %.8f\n", cos_sim_f(fused_out, cpu_row, dr));
    printf("Cos-sim fused vs col-major: %.8f\n", cos_sim_f(fused_out, cpu_col, dr));
    
    free(x); free(w); free(cpu_row); free(cpu_col); free(cublas_out); free(fused_out);
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_cublas); cudaFree(d_fused);
    wubu_cuda_destroy(handle, stream);
    return 0;
}
