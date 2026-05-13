/**
 * Test: Fused SSM Forward (parallel scan) vs step-by-step GPU
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

static double max_abs_diff_f32(const float *a, const float *b, int n) {
    double md = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > md) md = d;
    }
    return md;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *gguf_path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    printf("Loading: %s\n", gguf_path);

    // Load layer 0 weights
    gguf_ctx *ctx = gguf_open(gguf_path);
    if (!ctx) return 1;

    float *w_qkv = NULL, *w_gate = NULL, *w_beta = NULL, *w_alpha = NULL;
    float *w_dt = NULL, *w_a = NULL, *w_conv = NULL, *w_norm = NULL, *w_out = NULL;

    #define LOAD_T(name, buf, n) do { \
        gguf_tensor_info *t_ = gguf_find_tensor(ctx, name); \
        if (!t_) { fprintf(stderr, "MISSING %s\n", name); return 1; } \
        buf = (float*)malloc((n) * sizeof(float)); \
        gguf_read_tensor_f32(ctx, t_, buf, n); \
    } while(0)

    LOAD_T("blk.0.attn_qkv.weight", w_qkv, D_MODEL * CONV_DIM);
    LOAD_T("blk.0.attn_gate.weight", w_gate, D_MODEL * VALUE_DIM);
    LOAD_T("blk.0.ssm_beta.weight", w_beta, D_MODEL * DT_RANK);
    LOAD_T("blk.0.ssm_alpha.weight", w_alpha, D_MODEL * DT_RANK);
    LOAD_T("blk.0.ssm_dt.bias", w_dt, DT_RANK);
    LOAD_T("blk.0.ssm_a", w_a, DT_RANK);
    LOAD_T("blk.0.ssm_conv1d.weight", w_conv, CONV_KERNEL * CONV_DIM);
    LOAD_T("blk.0.ssm_norm.weight", w_norm, SSM_D_STATE);
    LOAD_T("blk.0.ssm_out.weight", w_out, VALUE_DIM * D_MODEL);
    gguf_close(ctx);

    cublasHandle_t handle;
    cudaStream_t stream;
    wubu_cuda_init(&handle, &stream);

    const int B = 1, T = 8, N = B * T;
    const int n_state = B * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;

    // Random input
    float *x = (float*)malloc(N * D_MODEL * sizeof(float));
    srand(42);
    for (int i = 0; i < N * D_MODEL; i++)
        x[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;

    // GPU weights
    float *d_w_qkv = wubu_cuda_alloc(D_MODEL * CONV_DIM * sizeof(float));
    float *d_w_gate = wubu_cuda_alloc(D_MODEL * VALUE_DIM * sizeof(float));
    float *d_w_beta = wubu_cuda_alloc(D_MODEL * DT_RANK * sizeof(float));
    float *d_w_alpha = wubu_cuda_alloc(D_MODEL * DT_RANK * sizeof(float));
    float *d_dt = wubu_cuda_alloc(DT_RANK * sizeof(float));
    float *d_a = wubu_cuda_alloc(DT_RANK * sizeof(float));
    float *d_conv = wubu_cuda_alloc(CONV_KERNEL * CONV_DIM * sizeof(float));
    float *d_norm = wubu_cuda_alloc(SSM_D_STATE * sizeof(float));
    float *d_out = wubu_cuda_alloc(VALUE_DIM * D_MODEL * sizeof(float));
    float *d_x = wubu_cuda_alloc(N * D_MODEL * sizeof(float));
    float *d_scratch = NULL;
    float *d_h_new = wubu_cuda_alloc(n_state * sizeof(float));
    float *d_cs_new = wubu_cuda_alloc(B * 3 * CONV_DIM * sizeof(float));
    float *d_o_new = wubu_cuda_alloc(N * D_MODEL * sizeof(float));

    wubu_cuda_to_device(w_qkv, d_w_qkv, D_MODEL * CONV_DIM * sizeof(float), stream);
    wubu_cuda_to_device(w_gate, d_w_gate, D_MODEL * VALUE_DIM * sizeof(float), stream);
    wubu_cuda_to_device(w_beta, d_w_beta, D_MODEL * DT_RANK * sizeof(float), stream);
    wubu_cuda_to_device(w_alpha, d_w_alpha, D_MODEL * DT_RANK * sizeof(float), stream);
    wubu_cuda_to_device(w_dt, d_dt, DT_RANK * sizeof(float), stream);
    wubu_cuda_to_device(w_a, d_a, DT_RANK * sizeof(float), stream);
    wubu_cuda_to_device(w_conv, d_conv, CONV_KERNEL * CONV_DIM * sizeof(float), stream);
    wubu_cuda_to_device(w_norm, d_norm, SSM_D_STATE * sizeof(float), stream);
    wubu_cuda_to_device(w_out, d_out, VALUE_DIM * D_MODEL * sizeof(float), stream);
    wubu_cuda_to_device(x, d_x, N * D_MODEL * sizeof(float), stream);
    cudaMemset(d_h_new, 0, n_state * sizeof(float));
    cudaMemset(d_cs_new, 0, B * 3 * CONV_DIM * sizeof(float));
    cudaStreamSynchronize(stream);

    // Allocate scratch
    size_t scratch_sz = wubu_cuda_ssm_forward_query_scratch(B, T);
    d_scratch = wubu_cuda_alloc(scratch_sz);

    // === Fused SSM forward ===
    double t0 = now_sec();
    wubu_cuda_ssm_forward(handle, stream, B, T, d_x,
        d_w_qkv, d_w_gate, d_w_beta, d_w_alpha, d_dt, d_a,
        d_conv, d_norm, d_out, d_h_new, d_cs_new, d_o_new, d_scratch);
    cudaStreamSynchronize(stream);
    double t_fused = now_sec() - t0;

    float *out_fused = (float*)malloc(N * D_MODEL * sizeof(float));
    cudaMemcpy(out_fused, d_o_new, N * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost);

    // === CPU reference ===
    ssm_layer_weights w;
    memset(&w, 0, sizeof(w));
    w.attn_qkv_weight = w_qkv;
    w.attn_gate_weight = w_gate;
    w.ssm_beta_weight = w_beta;
    w.ssm_alpha_weight = w_alpha;
    w.ssm_dt_bias = w_dt;
    w.ssm_a = w_a;
    w.ssm_conv1d_weight = w_conv;
    w.ssm_norm_weight = w_norm;
    w.ssm_out_weight = w_out;
    w.attn_norm_weight = (float*)malloc(D_MODEL * sizeof(float));
    w.post_attention_norm_weight = (float*)malloc(D_MODEL * sizeof(float));
    memset(w.attn_norm_weight, 0, D_MODEL * sizeof(float));
    memset(w.post_attention_norm_weight, 0, D_MODEL * sizeof(float));

    float *ssm_state_cpu = (float*)calloc(n_state, sizeof(float));
    float *conv_state_cpu = (float*)calloc(B * 3 * CONV_DIM, sizeof(float));
    float *out_cpu = (float*)malloc(N * D_MODEL * sizeof(float));

    t0 = now_sec();
    wubu_ssm_forward(x, B, T, &w, ssm_state_cpu, conv_state_cpu, out_cpu);
    double t_cpu = now_sec() - t0;

    // Compare
    printf("=== Fused SSM Forward vs CPU ===\n");
    double diff = max_abs_diff_f32(out_cpu, out_fused, N * D_MODEL);
    printf("Output max diff: %.10f\n", diff);
    printf("Fused: %.3f ms | CPU: %.3f ms\n", t_fused * 1000.0, t_cpu * 1000.0);

    // Timing with iteration loop
    cudaMemset(d_h_new, 0, n_state * sizeof(float));
    cudaMemset(d_cs_new, 0, B * 3 * CONV_DIM * sizeof(float));
    cudaStreamSynchronize(stream);
    t0 = now_sec();
    const int ITERS = 20;
    for (int i = 0; i < ITERS; i++) {
        wubu_cuda_ssm_forward(handle, stream, B, T, d_x,
            d_w_qkv, d_w_gate, d_w_beta, d_w_alpha, d_dt, d_a,
            d_conv, d_norm, d_out, d_h_new, d_cs_new, d_o_new, d_scratch);
    }
    cudaStreamSynchronize(stream);
    double t_avg = (now_sec() - t0) / ITERS;
    printf("Avg fused (x%d): %.3f ms (%.0f tok/s)\n", ITERS,
           t_avg * 1000.0, (double)N / t_avg);

    int pass = diff < 5e-3 ? 1 : 0;
    printf("\n=== %s (threshold 5e-3) ===\n", pass ? "PASS" : "FAIL");

    free(x); free(out_cpu); free(out_fused);
    free(ssm_state_cpu); free(conv_state_cpu);
    free(w.attn_norm_weight); free(w.post_attention_norm_weight);

    wubu_cuda_free(d_w_qkv); wubu_cuda_free(d_w_gate);
    wubu_cuda_free(d_w_beta); wubu_cuda_free(d_w_alpha);
    wubu_cuda_free(d_dt); wubu_cuda_free(d_a);
    wubu_cuda_free(d_conv); wubu_cuda_free(d_norm); wubu_cuda_free(d_out);
    wubu_cuda_free(d_x); wubu_cuda_free(d_scratch);
    wubu_cuda_free(d_h_new); wubu_cuda_free(d_cs_new); wubu_cuda_free(d_o_new);
    free(w_qkv); free(w_gate); free(w_beta); free(w_alpha);
    free(w_dt); free(w_a); free(w_conv); free(w_norm); free(w_out);

    wubu_cuda_destroy(handle, stream);
    return pass ? 0 : 1;
}
