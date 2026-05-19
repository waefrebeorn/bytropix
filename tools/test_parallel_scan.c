/**
 * Test: Parallel SSM Scan Kernel vs CPU Reference
 *
 * Launches the ssm_parallel_scan_kernel, compares against
 * the CPU wubu_ssm_forward() output, and reports max diff.
 */

#include "wubu_ssm.h"
#include "cuda_kernels.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define B 2
#define T 32

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float md = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > md) md = d;
    }
    return md;
}

static gguf_tensor_info* load_tensor(gguf_ctx *ctx, const char *name, float **buf, int64_t n) {
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "MISSING TENSOR: %s\n", name);
        return NULL;
    }
    *buf = (float*)malloc(n * sizeof(float));
    if (!*buf) { fprintf(stderr, "OOM for %s\n", name); return NULL; }
    int nr = gguf_read_tensor_f32(ctx, t, *buf, n);
    if (nr != n) {
        fprintf(stderr, "WARN: %s read %d elems, expected %ld\n", name, nr, (long)n);
    }
    return t;
}

int main(int argc, char **argv) {
    const char *gguf_path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    printf("Loading model: %s\n", gguf_path);
    
    // Load first SSM layer weights (layer 0) from GGUF
    ssm_layer_weights w_cpu;
    memset(&w_cpu, 0, sizeof(w_cpu));
    
    gguf_ctx *ctx = gguf_open(gguf_path);
    if (!ctx) { fprintf(stderr, "Failed to open GGUF\n"); return 1; }
    
    char name[256];
    int test_layer = 0;
    const int qkv_dim = CONV_DIM;  // 8192
    
    snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", test_layer);
    load_tensor(ctx, name, &w_cpu.attn_norm_weight, D_MODEL);
    
    snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", test_layer);
    load_tensor(ctx, name, &w_cpu.attn_qkv_weight, D_MODEL * qkv_dim);
    
    snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", test_layer);
    load_tensor(ctx, name, &w_cpu.attn_gate_weight, D_MODEL * VALUE_DIM);
    
    snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", test_layer);
    load_tensor(ctx, name, &w_cpu.ssm_conv1d_weight, CONV_KERNEL * CONV_DIM);
    
    snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", test_layer);
    load_tensor(ctx, name, &w_cpu.ssm_dt_bias, DT_RANK);
    
    snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", test_layer);
    load_tensor(ctx, name, &w_cpu.ssm_norm_weight, SSM_D_STATE);
    
    snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", test_layer);
    load_tensor(ctx, name, &w_cpu.ssm_out_weight, VALUE_DIM * D_MODEL);
    
    snprintf(name, sizeof(name), "blk.%d.ssm_a", test_layer);
    load_tensor(ctx, name, &w_cpu.ssm_a, DT_RANK);
    
    snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", test_layer);
    load_tensor(ctx, name, &w_cpu.ssm_alpha_weight, D_MODEL * DT_RANK);
    
    snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", test_layer);
    load_tensor(ctx, name, &w_cpu.ssm_beta_weight, D_MODEL * DT_RANK);
    
    snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", test_layer);
    load_tensor(ctx, name, &w_cpu.post_attention_norm_weight, D_MODEL);
    
    gguf_close(ctx);
    printf("Layer %d weights loaded.\n", test_layer);
    
    // Verify required weights loaded
    if (!w_cpu.attn_qkv_weight || !w_cpu.attn_gate_weight || !w_cpu.ssm_beta_weight ||
        !w_cpu.ssm_alpha_weight || !w_cpu.ssm_dt_bias || !w_cpu.ssm_a ||
        !w_cpu.ssm_conv1d_weight || !w_cpu.ssm_norm_weight || !w_cpu.ssm_out_weight) {
        fprintf(stderr, "Failed to load required weights\n");
        return 1;
    }
    
    // Initialize CUDA
    cublasHandle_t cublas_h;
    cudaStream_t stream;
    if (!wubu_cuda_init(&cublas_h, &stream)) {
        fprintf(stderr, "CUDA init failed\n");
        return 1;
    }
    
    const int N = B * T;
    const int N_states = B * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
    
    // Create random input
    float *x_cpu = (float*)malloc(N * D_MODEL * sizeof(float));
    float *output_cpu = (float*)malloc(N * D_MODEL * sizeof(float));
    float *output_gpu = (float*)malloc(N * D_MODEL * sizeof(float));
    float *ssm_state_cpu = (float*)calloc(N_states, sizeof(float));
    float *ssm_state_gpu = (float*)calloc(N_states, sizeof(float));
    float *conv_state_cpu = (float*)calloc(B * (CONV_KERNEL-1) * CONV_DIM, sizeof(float));
    float *conv_state_gpu = (float*)calloc(B * (CONV_KERNEL-1) * CONV_DIM, sizeof(float));
    
    // Random input
    srand(42);
    for (int i = 0; i < N * D_MODEL; i++) {
        x_cpu[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    
    // ===== CPU reference =====
    printf("\n--- CPU forward pass ---\n");
    double t0 = now_sec();
    wubu_ssm_forward(x_cpu, B, T, &w_cpu, ssm_state_cpu, conv_state_cpu, output_cpu, NULL, NULL);
    double t_cpu = now_sec() - t0;
    printf("CPU: %.3f ms\n", t_cpu * 1000.0);
    
    // ===== Allocate GPU buffers =====
    float *d_w_qkv     = wubu_cuda_alloc(D_MODEL * CONV_DIM * sizeof(float));
    float *d_w_gate    = wubu_cuda_alloc(D_MODEL * VALUE_DIM * sizeof(float));
    float *d_w_beta    = wubu_cuda_alloc(D_MODEL * DT_RANK * sizeof(float));
    float *d_w_alpha   = wubu_cuda_alloc(D_MODEL * DT_RANK * sizeof(float));
    float *d_dt_bias   = wubu_cuda_alloc(DT_RANK * sizeof(float));
    float *d_ssm_a     = wubu_cuda_alloc(DT_RANK * sizeof(float));
    float *d_conv1d    = wubu_cuda_alloc(CONV_KERNEL * CONV_DIM * sizeof(float));
    float *d_norm_w    = wubu_cuda_alloc(SSM_D_STATE * sizeof(float));
    float *d_out_w     = wubu_cuda_alloc(VALUE_DIM * D_MODEL * sizeof(float));
    
    float *d_x         = wubu_cuda_alloc(N * D_MODEL * sizeof(float));
    float *d_output    = wubu_cuda_alloc(N * D_MODEL * sizeof(float));
    float *d_qkv       = wubu_cuda_alloc(N * CONV_DIM * sizeof(float));
    float *d_z         = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));
    float *d_beta_raw  = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    float *d_alpha_raw = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    float *d_beta_sig  = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    float *d_gate      = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    float *d_alpha_bi  = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    float *d_conv_in   = wubu_cuda_alloc(B * (T + 3) * CONV_DIM * sizeof(float));
    float *d_conv_out  = wubu_cuda_alloc(N * CONV_DIM * sizeof(float));
    float *d_q_conv    = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
    float *d_k_conv    = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
    float *d_v_conv    = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));
    float *d_q_norm    = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
    float *d_k_norm    = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
    float *d_delta_out = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));
    float *d_z_silu    = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));
    float *d_ssm_state = wubu_cuda_alloc(N_states * sizeof(float));
    float *d_conv_state= wubu_cuda_alloc(B * 3 * CONV_DIM * sizeof(float));
    
    if (!d_w_qkv || !d_x || !d_qkv || !d_conv_out || !d_delta_out) {
        fprintf(stderr, "GPU alloc failed\n"); return 1;
    }
    
    // Copy weights and input to GPU
    wubu_cuda_to_device(w_cpu.attn_qkv_weight,  d_w_qkv,  D_MODEL * CONV_DIM * sizeof(float), stream);
    wubu_cuda_to_device(w_cpu.attn_gate_weight, d_w_gate, D_MODEL * VALUE_DIM * sizeof(float), stream);
    wubu_cuda_to_device(w_cpu.ssm_beta_weight,  d_w_beta, D_MODEL * DT_RANK * sizeof(float), stream);
    wubu_cuda_to_device(w_cpu.ssm_alpha_weight, d_w_alpha,D_MODEL * DT_RANK * sizeof(float), stream);
    wubu_cuda_to_device(w_cpu.ssm_dt_bias,      d_dt_bias, DT_RANK * sizeof(float), stream);
    wubu_cuda_to_device(w_cpu.ssm_a,            d_ssm_a,  DT_RANK * sizeof(float), stream);
    wubu_cuda_to_device(w_cpu.ssm_conv1d_weight,d_conv1d, CONV_KERNEL * CONV_DIM * sizeof(float), stream);
    wubu_cuda_to_device(w_cpu.ssm_norm_weight,  d_norm_w, SSM_D_STATE * sizeof(float), stream);
    wubu_cuda_to_device(w_cpu.ssm_out_weight,   d_out_w,  VALUE_DIM * D_MODEL * sizeof(float), stream);
    wubu_cuda_to_device(x_cpu, d_x, N * D_MODEL * sizeof(float), stream);
    
    // Zero states
    cudaMemsetAsync(d_ssm_state, 0, N_states * sizeof(float), stream);
    cudaMemsetAsync(d_conv_state, 0, B * 3 * CONV_DIM * sizeof(float), stream);
    cudaMemsetAsync(d_conv_in, 0, B * (T+3) * CONV_DIM * sizeof(float), stream);
    cudaStreamSynchronize(stream);
    
    // ===== OLD METHOD: host-loop delta_net_step =====
    printf("\n--- GPU: per-token host-loop (old) ---\n");
    double t1 = now_sec();
    
    // Step 1-4: projections (cuBLAS + elementwise)
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_w_qkv, CONV_DIM, d_qkv, 1.0f, 0.0f);
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_w_gate, VALUE_DIM, d_z, 1.0f, 0.0f);
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_w_beta, DT_RANK, d_beta_raw, 1.0f, 0.0f);
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_w_alpha, DT_RANK, d_alpha_raw, 1.0f, 0.0f);
    wubu_cuda_sigmoid(N * DT_RANK, d_beta_raw, d_beta_sig, stream);
    wubu_cuda_add_bias(N, DT_RANK, d_alpha_raw, d_dt_bias, d_alpha_bi, stream);
    wubu_cuda_softplus(N * DT_RANK, d_alpha_bi, d_gate, stream);
    wubu_cuda_mul_by_scalar(N, DT_RANK, d_gate, d_ssm_a, d_gate, stream);
    
    // Step 5-7: conv
    for (int b = 0; b < B; b++) {
        cudaMemcpyAsync(d_conv_in + b * (T+3) * CONV_DIM + 3 * CONV_DIM,
                       d_qkv + b * T * CONV_DIM,
                       T * CONV_DIM * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
    wubu_cuda_conv1d(B, T, CONV_DIM, CONV_KERNEL, d_conv_in, d_conv1d, d_conv_out, stream);
    wubu_cuda_silu(N * CONV_DIM, d_conv_out, d_conv_out, stream);
    wubu_cuda_split_qkv(N, KEY_DIM, VALUE_DIM, d_conv_out, d_q_conv, d_k_conv, d_v_conv, stream);
    wubu_cuda_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, d_q_conv, 1e-12f, d_q_norm, stream);
    wubu_cuda_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, d_k_conv, 1e-12f, d_k_norm, stream);
    
    // Step 9: host-loop recurrence (the bottleneck)
    int repeat_factor = SSM_V_HEADS / SSM_K_HEADS;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int s = b * T + t;
            float beta_host[DT_RANK], gate_host[DT_RANK];
            cudaMemcpyAsync(beta_host, d_beta_sig + s * DT_RANK,
                           DT_RANK * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(gate_host, d_gate + s * DT_RANK,
                           DT_RANK * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            
            for (int vh = 0; vh < SSM_V_HEADS; vh++) {
                int kh = vh / repeat_factor;
                float *d_q_vh = d_q_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                float *d_k_vh = d_k_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                float *d_v_vh = d_v_conv + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                float *d_h = d_ssm_state + (vh * SSM_D_STATE * SSM_D_STATE);
                float *d_out_vh = d_delta_out + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                wubu_cuda_delta_net_step(d_h, d_q_vh, d_k_vh, d_v_vh,
                                         gate_host[kh], beta_host[kh], d_out_vh, stream);
            }
        }
    }
    cudaStreamSynchronize(stream);
    
    wubu_cuda_silu(N * VALUE_DIM, d_z, d_z_silu, stream);
    wubu_cuda_gated_norm(B, T, SSM_V_HEADS, SSM_D_STATE, d_delta_out, d_norm_w, d_z_silu, stream);
    wubu_cuda_matmul(cublas_h, d_delta_out, N, VALUE_DIM, d_out_w, D_MODEL, d_output, 1.0f, 0.0f);
    cudaStreamSynchronize(stream);
    double t_gpu_old = now_sec() - t1;
    
    float *delta_out_old = (float*)malloc(N * VALUE_DIM * sizeof(float));
    float *ssm_state_old = (float*)malloc(N_states * sizeof(float));
    float *output_old = (float*)malloc(N * D_MODEL * sizeof(float));
    cudaMemcpy(delta_out_old, d_delta_out, N * VALUE_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ssm_state_old, d_ssm_state, N_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_old, d_output, N * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Old GPU: %.3f ms\n", t_gpu_old * 1000.0);
    printf("  CPU vs Old output:  max diff = %.6f\n", max_abs_diff(output_cpu, output_old, N * D_MODEL));
    printf("  CPU vs Old state:   max diff = %.6f\n", max_abs_diff(ssm_state_cpu, ssm_state_old, N_states));
    
    // ===== NEW METHOD: parallel scan kernel =====
    printf("\n--- GPU: parallel scan (new) ---\n");
    
    // Reset states
    cudaMemsetAsync(d_ssm_state, 0, N_states * sizeof(float), stream);
    cudaMemsetAsync(d_delta_out, 0, N * VALUE_DIM * sizeof(float), stream);
    cudaStreamSynchronize(stream);
    
    double t2 = now_sec();
    
    // Steps 1-4: projections (same)
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_w_qkv, CONV_DIM, d_qkv, 1.0f, 0.0f);
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_w_gate, VALUE_DIM, d_z, 1.0f, 0.0f);
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_w_beta, DT_RANK, d_beta_raw, 1.0f, 0.0f);
    wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_w_alpha, DT_RANK, d_alpha_raw, 1.0f, 0.0f);
    wubu_cuda_sigmoid(N * DT_RANK, d_beta_raw, d_beta_sig, stream);
    wubu_cuda_add_bias(N, DT_RANK, d_alpha_raw, d_dt_bias, d_alpha_bi, stream);
    wubu_cuda_softplus(N * DT_RANK, d_alpha_bi, d_gate, stream);
    wubu_cuda_mul_by_scalar(N, DT_RANK, d_gate, d_ssm_a, d_gate, stream);
    
    // Steps 5-7: conv
    for (int b = 0; b < B; b++) {
        cudaMemcpyAsync(d_conv_in + b * (T+3) * CONV_DIM + 3 * CONV_DIM,
                       d_qkv + b * T * CONV_DIM,
                       T * CONV_DIM * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
    wubu_cuda_conv1d(B, T, CONV_DIM, CONV_KERNEL, d_conv_in, d_conv1d, d_conv_out, stream);
    wubu_cuda_silu(N * CONV_DIM, d_conv_out, d_conv_out, stream);
    wubu_cuda_split_qkv(N, KEY_DIM, VALUE_DIM, d_conv_out, d_q_conv, d_k_conv, d_v_conv, stream);
    wubu_cuda_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, d_q_conv, 1e-12f, d_q_norm, stream);
    wubu_cuda_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, d_k_conv, 1e-12f, d_k_norm, stream);
    
    // NEW: parallel scan in a single kernel launch
    wubu_cuda_ssm_parallel_scan(B, T,
        d_q_norm, d_k_norm, d_v_conv,
        d_gate, d_beta_sig,
        d_ssm_state, d_delta_out,
        stream);
    cudaStreamSynchronize(stream);
    
    // Steps 12-14: gated norm + output projection
    wubu_cuda_silu(N * VALUE_DIM, d_z, d_z_silu, stream);
    wubu_cuda_gated_norm(B, T, SSM_V_HEADS, SSM_D_STATE, d_delta_out, d_norm_w, d_z_silu, stream);
    wubu_cuda_matmul(cublas_h, d_delta_out, N, VALUE_DIM, d_out_w, D_MODEL, d_output, 1.0f, 0.0f);
    cudaStreamSynchronize(stream);
    double t_gpu_new = now_sec() - t2;
    
    // Copy results
    float *delta_out_new = (float*)malloc(N * VALUE_DIM * sizeof(float));
    float *ssm_state_new = (float*)malloc(N_states * sizeof(float));
    float *output_new = (float*)malloc(N * D_MODEL * sizeof(float));
    cudaMemcpy(delta_out_new, d_delta_out, N * VALUE_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ssm_state_new, d_ssm_state, N_states * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_new, d_output, N * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("New GPU: %.3f ms\n", t_gpu_new * 1000.0);
    
    // ===== Compare =====
    printf("\n=== Comparison ===\n");
    float cpu_vs_new_out  = max_abs_diff(output_cpu, output_new, N * D_MODEL);
    float cpu_vs_new_state= max_abs_diff(ssm_state_cpu, ssm_state_new, N_states);
    printf("CPU vs New (output):  max diff = %.6f\n", cpu_vs_new_out);
    printf("CPU vs New (state):   max diff = %.6f\n", cpu_vs_new_state);
    
    float old_vs_new_delta = max_abs_diff(delta_out_old, delta_out_new, N * VALUE_DIM);
    float old_vs_new_state = max_abs_diff(ssm_state_old, ssm_state_new, N_states);
    printf("Old vs New (delta):   max diff = %.6f\n", old_vs_new_delta);
    printf("Old vs New (state):   max diff = %.6f\n", old_vs_new_state);
    
    printf("\n--- Timing ---\n");
    printf("CPU (B=%d T=%d):        %.3f ms\n", B, T, t_cpu * 1000.0);
    printf("GPU old (host-loop):    %.3f ms  (scan part only: %.1f%% of GPU)\n", 
           t_gpu_old * 1000.0, 100.0);
    printf("GPU new (parallel scan): %.3f ms\n", t_gpu_new * 1000.0);
    
    // Cleanup
    free(x_cpu); free(output_cpu); free(output_gpu);
    free(ssm_state_cpu); free(ssm_state_gpu);
    free(conv_state_cpu); free(conv_state_gpu);
    free(delta_out_old); free(delta_out_new);
    free(ssm_state_old); free(ssm_state_new);
    free(output_old); free(output_new);
    
    #define GPU_FREE(p) do { if (p) wubu_cuda_free(p); } while(0)
    GPU_FREE(d_w_qkv); GPU_FREE(d_w_gate); GPU_FREE(d_w_beta); GPU_FREE(d_w_alpha);
    GPU_FREE(d_dt_bias); GPU_FREE(d_ssm_a); GPU_FREE(d_conv1d);
    GPU_FREE(d_norm_w); GPU_FREE(d_out_w);
    GPU_FREE(d_x); GPU_FREE(d_output); GPU_FREE(d_qkv); GPU_FREE(d_z);
    GPU_FREE(d_beta_raw); GPU_FREE(d_alpha_raw); GPU_FREE(d_beta_sig);
    GPU_FREE(d_gate); GPU_FREE(d_alpha_bi);
    GPU_FREE(d_conv_in); GPU_FREE(d_conv_out);
    GPU_FREE(d_q_conv); GPU_FREE(d_k_conv); GPU_FREE(d_v_conv);
    GPU_FREE(d_q_norm); GPU_FREE(d_k_norm);
    GPU_FREE(d_delta_out); GPU_FREE(d_z_silu);
    GPU_FREE(d_ssm_state); GPU_FREE(d_conv_state);
    
    free(w_cpu.attn_qkv_weight); free(w_cpu.attn_gate_weight);
    free(w_cpu.ssm_beta_weight); free(w_cpu.ssm_alpha_weight);
    free(w_cpu.ssm_dt_bias); free(w_cpu.ssm_a);
    free(w_cpu.ssm_conv1d_weight); free(w_cpu.ssm_norm_weight);
    free(w_cpu.ssm_out_weight);
    free(w_cpu.attn_norm_weight); free(w_cpu.post_attention_norm_weight);
    
    wubu_cuda_destroy(cublas_h, stream);
    
    float pass = (cpu_vs_new_out < 5e-3f && cpu_vs_new_state < 5e-3f) ? 1.0f : 0.0f;
    printf("\n=== %s (max diff threshold: 5e-3) ===\n", pass ? "PASS" : "FAIL");
    
    return pass ? 0 : 1;
}
