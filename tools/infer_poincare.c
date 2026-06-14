/**
 * infer_poincare.c — Poincaré SSM inference engine
 * Loads one SSM layer, runs GPU Poincaré forward, benchmarks.
 */
#include "bench.h"
#include "cuda_kernels.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "wubu_core_dumps.h"

int main(int argc, char **argv) {
    wubu_disable_core_dumps();
    const char *path = argc > 1 ? argv[1] : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    int layer = argc > 2 ? atoi(argv[2]) : 0;
    int B = 1, T = 4;
    float R = argc > 3 ? atof(argv[3]) : 0.956f;
    
    printf("=== Poincaré SSM Inference Engine ===\n");
    double t0 = now_sec();
    
    // CUDA init
    cublasHandle_t cublas_h;
    cudaStream_t stream;
    if (!wubu_cuda_init(&cublas_h, &stream)) return 1;
    
    // Load + buffer GGUF
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    
    // Load one SSM layer weights to GPU
    gpu_ssm_weights gw;
    if (!gpu_load_ssm_layer(ctx, layer, &gw, stream)) return 1;
    cudaStreamSynchronize(stream);
    
    // Allocate GPU buffers
    int N = B * T;
    float *d_x = wubu_cuda_alloc(N * D_MODEL * sizeof(float));
    float *d_out = wubu_cuda_alloc(N * D_MODEL * sizeof(float));
    float *d_ssm_state = wubu_cuda_alloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
    float *d_conv_state = wubu_cuda_alloc((CONV_KERNEL-1) * CONV_DIM * sizeof(float));
    cudaMemset(d_ssm_state, 0, SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
    
    // Scratch
    int qkv_dim = KEY_DIM * 2 + VALUE_DIM;
    float *d_qkv      = wubu_cuda_alloc(N * qkv_dim * sizeof(float));
    float *d_z        = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));
    float *d_beta     = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    float *d_alpha    = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    float *d_beta_sig = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    float *d_alpha_bi = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    float *d_gate     = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    float *d_conv_in  = wubu_cuda_alloc(B * (T + CONV_KERNEL - 1) * CONV_DIM * sizeof(float));
    float *d_conv_out = wubu_cuda_alloc(N * CONV_DIM * sizeof(float));
    float *d_q_conv   = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
    float *d_k_conv   = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
    float *d_v_conv   = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));
    float *d_q_norm   = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
    float *d_k_norm   = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
    float *d_delta    = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));
    float *d_z_silu   = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));
    
    // Upload test input
    float *x = (float *)malloc(N * D_MODEL * sizeof(float));
    for (int i = 0; i < N * D_MODEL; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
    cudaMemcpy(d_x, x, N * D_MODEL * sizeof(float), cudaMemcpyHostToDevice);
    
    printf("  Init: %.2fs\\n", now_sec() - t0);
    
    // Warmup
    gpu_poincare_ssm_forward(cublas_h, stream, d_x, B, T,
        gw.d_attn_qkv, gw.d_attn_gate, gw.d_ssm_beta, gw.d_ssm_alpha,
        gw.d_ssm_dt_bias, gw.d_ssm_a, gw.d_ssm_conv1d, gw.d_ssm_norm, gw.d_ssm_out,
        d_ssm_state, d_conv_state, d_out,
        d_qkv, d_z, d_beta, d_alpha, d_beta_sig, d_alpha_bi, d_gate,
        d_conv_in, d_conv_out, d_q_conv, d_k_conv, d_v_conv,
        d_q_norm, d_k_norm, d_delta, d_z_silu, R);
    
    // Benchmark
    int iters = 100;
    cudaDeviceSynchronize();
    double t1 = now_sec();
    for (int i = 0; i < iters; i++) {
        cudaMemset(d_ssm_state, 0, SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
        gpu_poincare_ssm_forward(cublas_h, stream, d_x, B, T,
            gw.d_attn_qkv, gw.d_attn_gate, gw.d_ssm_beta, gw.d_ssm_alpha,
            gw.d_ssm_dt_bias, gw.d_ssm_a, gw.d_ssm_conv1d, gw.d_ssm_norm, gw.d_ssm_out,
            d_ssm_state, d_conv_state, d_out,
            d_qkv, d_z, d_beta, d_alpha, d_beta_sig, d_alpha_bi, d_gate,
            d_conv_in, d_conv_out, d_q_conv, d_k_conv, d_v_conv,
            d_q_norm, d_k_norm, d_delta, d_z_silu, R);
    }
    cudaDeviceSynchronize();
    double t_total = now_sec() - t1;
    
    printf("  Poincaré SSM forward: avg %.3f ms (%.0f tok/s, %d iters)\\n",
           t_total / iters * 1000, B * T * iters / t_total, iters);
    
    // Check output
    float *h_out = (float *)malloc(N * D_MODEL * sizeof(float));
    cudaMemcpy(h_out, d_out, N * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost);
    float min_v = 1e30, max_v = -1e30; int nan_c = 0;
    for (int i = 0; i < N * D_MODEL; i++) {
        if (h_out[i] < min_v) min_v = h_out[i];
        if (h_out[i] > max_v) max_v = h_out[i];
        if (isnan(h_out[i])) nan_c++;
    }
    printf("  Output range: [%.4f, %.4f] | NaN: %d\\n", min_v, max_v, nan_c);
    
    // Cleanup
    free(x); free(h_out);
    wubu_cuda_free(d_x); wubu_cuda_free(d_out);
    wubu_cuda_free(d_ssm_state); wubu_cuda_free(d_conv_state);
    wubu_cuda_free(d_qkv); wubu_cuda_free(d_z);
    wubu_cuda_free(d_beta); wubu_cuda_free(d_alpha);
    wubu_cuda_free(d_beta_sig); wubu_cuda_free(d_alpha_bi);
    wubu_cuda_free(d_gate); wubu_cuda_free(d_conv_in);
    wubu_cuda_free(d_conv_out); wubu_cuda_free(d_q_conv);
    wubu_cuda_free(d_k_conv); wubu_cuda_free(d_v_conv);
    wubu_cuda_free(d_q_norm); wubu_cuda_free(d_k_norm);
    wubu_cuda_free(d_delta); wubu_cuda_free(d_z_silu);
    gpu_free_ssm_weights(&gw);
    gguf_close(ctx);
    wubu_cuda_destroy(cublas_h, stream);
    
    printf("=== Poincaré Inference PASS ===\\n");
    return 0;
}
