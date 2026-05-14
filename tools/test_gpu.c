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

/**
 * Phase 2.5 GPU Test
 *
 * Loads 1 SSM layer from GGUF, runs forward pass on GPU using cuBLAS + CUDA kernels,
 * compares output with CPU forward pass. Reports max diff.
 *
 * If max diff < 1e-4: GPU kernels are correct.
 */

// Forward declarations for CUDA kernels used in GQA test
// (Used via wubu_cuda_gqa_forward in cuda_kernels.o)

// Test configuration
#define B 1
#define T 4

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

static float max_abs_val(const float *a, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = fabsf(a[i]);
        if (v > m) m = v;
    }
    return m;
}

// Check if a tensor exists and log its info
static int check_tensor(gguf_ctx *ctx, const char *name) {
    gguf_tensor_info *t = gguf_find_tensor(ctx, name);
    if (!t) {
        printf("  MISSING: %s\n", name);
        return 0;
    }
    printf("  FOUND:   %s [", name);
    for (int i = 0; i < t->n_dims; i++)
        printf("%s%ld", i ? "," : "", (long)t->dims[i]);
    printf("] type=%d\n", t->ggml_type);
    return 1;
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] 
        : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    int test_layer = argc > 2 ? atoi(argv[2]) : 0;
    int run_gqa = argc > 3 ? atoi(argv[3]) : 0;  // 0=SSM, 1=GQA

    printf("=== WuBuText AI — Phase 2.5 GPU Test ===\n");
    printf("Model: %s\n", model_path);
    printf("Layer: %d (%s)\n", test_layer, run_gqa ? "GQA" : "SSM");
    printf("B=%d, T=%d\n", B, T);

    // ========== Init CUDA ==========
    cublasHandle_t cublas_h;
    cudaStream_t stream;
    if (!wubu_cuda_init(&cublas_h, &stream)) {
        fprintf(stderr, "CUDA init failed\n");
        return 1;
    }
    printf("CUDA init OK\n");

    // ========== Check model file ==========
    FILE *fm = fopen(model_path, "rb");
    if (!fm) {
        fprintf(stderr, "Cannot open %s\n", model_path);
        return 1;
    }
    fclose(fm);

    // ========== Load one SSM layer from GGUF ==========
    printf("Loading layer %d from GGUF...\n", test_layer);
    gguf_ctx *ctx = gguf_open(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to open %s\n", model_path);
        return 1;
    }

    int qkv_dim = KEY_DIM * 2 + VALUE_DIM;  // 8192

    if (!run_gqa) {
        // ---- SSM layer ----
        ssm_layer_weights w_cpu;
        memset(&w_cpu, 0, sizeof(w_cpu));

        char name[256];
        gguf_tensor_info *t;

        snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", test_layer);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
        w_cpu.attn_qkv_weight = (float *)malloc(D_MODEL * qkv_dim * sizeof(float));
        int nread = gguf_read_tensor_f32(ctx, t, w_cpu.attn_qkv_weight, D_MODEL * qkv_dim);
        printf("  attn_qkv read %d elems (expected %d)\n", nread, D_MODEL * qkv_dim);

        snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", test_layer);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
        w_cpu.attn_gate_weight = (float *)malloc(D_MODEL * VALUE_DIM * sizeof(float));
        gguf_read_tensor_f32(ctx, t, w_cpu.attn_gate_weight, D_MODEL * VALUE_DIM);

        snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", test_layer);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
        w_cpu.ssm_beta_weight = (float *)malloc(D_MODEL * DT_RANK * sizeof(float));
        gguf_read_tensor_f32(ctx, t, w_cpu.ssm_beta_weight, D_MODEL * DT_RANK);

        snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", test_layer);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
        w_cpu.ssm_alpha_weight = (float *)malloc(D_MODEL * DT_RANK * sizeof(float));
        gguf_read_tensor_f32(ctx, t, w_cpu.ssm_alpha_weight, D_MODEL * DT_RANK);

        snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", test_layer);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
        w_cpu.ssm_dt_bias = (float *)malloc(DT_RANK * sizeof(float));
        gguf_read_tensor_f32(ctx, t, w_cpu.ssm_dt_bias, DT_RANK);

        snprintf(name, sizeof(name), "blk.%d.ssm_a", test_layer);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
        w_cpu.ssm_a = (float *)malloc(DT_RANK * sizeof(float));
        gguf_read_tensor_f32(ctx, t, w_cpu.ssm_a, DT_RANK);

        snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", test_layer);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
        w_cpu.ssm_conv1d_weight = (float *)malloc(CONV_KERNEL * CONV_DIM * sizeof(float));
        gguf_read_tensor_f32(ctx, t, w_cpu.ssm_conv1d_weight, CONV_KERNEL * CONV_DIM);

        snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", test_layer);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
        w_cpu.ssm_norm_weight = (float *)malloc(SSM_D_STATE * sizeof(float));
        gguf_read_tensor_f32(ctx, t, w_cpu.ssm_norm_weight, SSM_D_STATE);

        snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", test_layer);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
        w_cpu.ssm_out_weight = (float *)malloc(VALUE_DIM * D_MODEL * sizeof(float));
        gguf_read_tensor_f32(ctx, t, w_cpu.ssm_out_weight, VALUE_DIM * D_MODEL);

        gguf_close(ctx);
        printf("SSM weights loaded (9 tensors)\n");

        // ========== Input data ==========
        const int N = B * T;
        float *x_cpu = (float *)malloc(N * D_MODEL * sizeof(float));
        srand(42);
        for (int i = 0; i < N * D_MODEL; i++) {
            x_cpu[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }

        // ========== CPU forward pass ==========
        float *output_cpu = (float *)calloc(N * D_MODEL, sizeof(float));
        float *ssm_state_cpu = (float *)calloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE, sizeof(float));
        float *conv_state_cpu = (float *)calloc((CONV_KERNEL - 1) * CONV_DIM, sizeof(float));

        double t0 = now_sec();
        wubu_ssm_forward(x_cpu, B, T, &w_cpu, ssm_state_cpu, conv_state_cpu, output_cpu);
        double t_cpu = now_sec() - t0;
        printf("CPU forward pass: %.3f ms\n", t_cpu * 1000.0);
        printf("CPU output[0:8]:");
        for (int i = 0; i < 8; i++) printf(" %+.6f", output_cpu[i]);
        printf("\n");
        printf("CPU output max val: %.6f\n", max_abs_val(output_cpu, N * D_MODEL));

        // ========== GPU forward pass ==========
        printf("Running GPU forward pass...\n");

        // Allocate GPU memory for weights
        float *d_attn_qkv    = wubu_cuda_alloc(D_MODEL * qkv_dim * sizeof(float));
        float *d_attn_gate   = wubu_cuda_alloc(D_MODEL * VALUE_DIM * sizeof(float));
        float *d_ssm_beta    = wubu_cuda_alloc(D_MODEL * DT_RANK * sizeof(float));
        float *d_ssm_alpha   = wubu_cuda_alloc(D_MODEL * DT_RANK * sizeof(float));
        float *d_ssm_dt_bias = wubu_cuda_alloc(DT_RANK * sizeof(float));
        float *d_ssm_a       = wubu_cuda_alloc(DT_RANK * sizeof(float));
        float *d_ssm_conv1d  = wubu_cuda_alloc(CONV_KERNEL * CONV_DIM * sizeof(float));
        float *d_ssm_norm    = wubu_cuda_alloc(SSM_D_STATE * sizeof(float));
        float *d_ssm_out     = wubu_cuda_alloc(VALUE_DIM * D_MODEL * sizeof(float));

        if (!d_attn_qkv || !d_attn_gate || !d_ssm_beta || !d_ssm_alpha ||
            !d_ssm_dt_bias || !d_ssm_a || !d_ssm_conv1d || !d_ssm_norm || !d_ssm_out) {
            fprintf(stderr, "GPU weight allocation failed\n");
            goto cleanup;
        }

        wubu_cuda_to_device(w_cpu.attn_qkv_weight, d_attn_qkv, D_MODEL * qkv_dim * sizeof(float), stream);
        wubu_cuda_to_device(w_cpu.attn_gate_weight, d_attn_gate, D_MODEL * VALUE_DIM * sizeof(float), stream);
        wubu_cuda_to_device(w_cpu.ssm_beta_weight, d_ssm_beta, D_MODEL * DT_RANK * sizeof(float), stream);
        wubu_cuda_to_device(w_cpu.ssm_alpha_weight, d_ssm_alpha, D_MODEL * DT_RANK * sizeof(float), stream);
        wubu_cuda_to_device(w_cpu.ssm_dt_bias, d_ssm_dt_bias, DT_RANK * sizeof(float), stream);
        wubu_cuda_to_device(w_cpu.ssm_a, d_ssm_a, DT_RANK * sizeof(float), stream);
        wubu_cuda_to_device(w_cpu.ssm_conv1d_weight, d_ssm_conv1d, CONV_KERNEL * CONV_DIM * sizeof(float), stream);
        wubu_cuda_to_device(w_cpu.ssm_norm_weight, d_ssm_norm, SSM_D_STATE * sizeof(float), stream);
        wubu_cuda_to_device(w_cpu.ssm_out_weight, d_ssm_out, VALUE_DIM * D_MODEL * sizeof(float), stream);

        // GPU buffers
        float *d_x           = wubu_cuda_alloc(N * D_MODEL * sizeof(float));
        float *d_output      = wubu_cuda_alloc(N * D_MODEL * sizeof(float));
        float *d_ssm_state   = wubu_cuda_alloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
        float *d_conv_state  = wubu_cuda_alloc((CONV_KERNEL - 1) * CONV_DIM * sizeof(float));
        float *d_qkv         = wubu_cuda_alloc(N * qkv_dim * sizeof(float));
        float *d_z           = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));
        float *d_beta        = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
        float *d_alpha       = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
        float *d_beta_sig    = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
        float *d_alpha_bi    = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
        float *d_gate        = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
        float *d_conv_input  = wubu_cuda_alloc(B * (T + CONV_KERNEL - 1) * CONV_DIM * sizeof(float));
        float *d_conv_out    = wubu_cuda_alloc(N * CONV_DIM * sizeof(float));
        float *d_q_conv      = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
        float *d_k_conv      = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
        float *d_v_conv      = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));
        float *d_q_norm      = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
        float *d_k_norm      = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
        float *d_delta_out   = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));
        float *d_z_silu      = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));

        if (!d_x || !d_output || !d_ssm_state || !d_conv_state ||
            !d_qkv || !d_z || !d_beta || !d_alpha || !d_beta_sig || !d_alpha_bi || !d_gate ||
            !d_conv_input || !d_conv_out ||
            !d_q_conv || !d_k_conv || !d_v_conv || !d_q_norm || !d_k_norm ||
            !d_delta_out || !d_z_silu) {
            fprintf(stderr, "GPU buffer allocation failed\n");
            goto cleanup;
        }

        // Zero states
        cudaMemsetAsync(d_ssm_state, 0, SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float), stream);
        cudaMemsetAsync(d_conv_state, 0, (CONV_KERNEL - 1) * CONV_DIM * sizeof(float), stream);
        wubu_cuda_to_device(x_cpu, d_x, N * D_MODEL * sizeof(float), stream);

        cudaStreamSynchronize(stream);
        double t1 = now_sec();

        // ===== Step 1: QKV projection =====
        // x[N,2048] @ attn_qkv[2048,8192] -> qkv[N,8192]
        wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_qkv, qkv_dim, d_qkv, 1.0f, 0.0f);

        // ===== Step 2: z gate projection =====
        // x[N,2048] @ attn_gate[2048,4096] -> z[N,4096]
        wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_gate, VALUE_DIM, d_z, 1.0f, 0.0f);

        // ===== Step 3: beta/alpha projections =====
        wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_ssm_beta, DT_RANK, d_beta, 1.0f, 0.0f);
        wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_ssm_alpha, DT_RANK, d_alpha, 1.0f, 0.0f);

        // ===== Step 4: beta = sigmoid(beta_raw), gate = softplus(alpha + dt_bias) * ssm_a =====
        wubu_cuda_sigmoid(N * DT_RANK, d_beta, d_beta_sig, stream);

        // alpha_biased = alpha + dt_bias (broadcast)
        wubu_cuda_add_bias(N, DT_RANK, d_alpha, d_ssm_dt_bias, d_alpha_bi, stream);

        // gate = softplus(alpha_biased)
        wubu_cuda_softplus(N * DT_RANK, d_alpha_bi, d_gate, stream);

        // gate *= ssm_a (broadcast)
        wubu_cuda_mul_by_scalar(N, DT_RANK, d_gate, d_ssm_a, d_gate, stream);

        // ===== Step 5: Convolution =====
        // Build conv_input = [conv_state | qkv]
        // First CONV_KERNEL-1 positions are conv_state (zeros), then qkv_all
        for (int b = 0; b < B; b++) {
            cudaMemcpyAsync(d_conv_input + b * (T + CONV_KERNEL - 1) * CONV_DIM + (CONV_KERNEL - 1) * CONV_DIM,
                           d_qkv + b * T * CONV_DIM,
                           T * CONV_DIM * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream);
        }

        // Convolution + SiLU
        wubu_cuda_conv1d(B, T, CONV_DIM, CONV_KERNEL, d_conv_input, d_ssm_conv1d, d_conv_out, stream);
        wubu_cuda_silu(N * CONV_DIM, d_conv_out, d_conv_out, stream);

        // Update conv_state: last CONV_KERNEL-1 elements
        for (int b = 0; b < B; b++) {
            cudaMemcpyAsync(d_conv_state + b * (CONV_KERNEL - 1) * CONV_DIM,
                           d_conv_input + (b * (T + CONV_KERNEL - 1) + T) * CONV_DIM,
                           (CONV_KERNEL - 1) * CONV_DIM * sizeof(float),
                           cudaMemcpyDeviceToDevice, stream);
        }

        // ===== Step 6: Split Q, K, V =====
        wubu_cuda_split_qkv(N, KEY_DIM, VALUE_DIM, d_conv_out,
                            d_q_conv, d_k_conv, d_v_conv, stream);

        // ===== Step 7: L2 Normalize Q and K =====
        wubu_cuda_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, d_q_conv, 1e-12f, d_q_norm, stream);
        wubu_cuda_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, d_k_conv, 1e-12f, d_k_norm, stream);

        // ===== Step 8: repeat_factor (implicit via delta_net_step) =====
        int repeat_factor = SSM_V_HEADS / SSM_K_HEADS;

        // ===== Step 9: Gated Delta Net recurrence =====
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                int s = b * T + t;

                // Get per-token beta and gate from GPU
                float beta_host[DT_RANK], gate_host[DT_RANK];
                cudaMemcpyAsync(beta_host, d_beta_sig + s * DT_RANK,
                               DT_RANK * sizeof(float), cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(gate_host, d_gate + s * DT_RANK,
                               DT_RANK * sizeof(float), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);

                for (int vh = 0; vh < SSM_V_HEADS; vh++) {
                    int kh = vh / repeat_factor;
                    float bg = beta_host[kh];
                    float gg = gate_host[kh];

                    float *d_q_vh = d_q_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                    float *d_k_vh = d_k_norm + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                    float *d_v_vh = d_v_conv + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                    float *d_h = d_ssm_state + (vh * SSM_D_STATE * SSM_D_STATE);
                    float *d_out_vh = d_delta_out + (s * SSM_V_HEADS + vh) * SSM_D_STATE;

                    wubu_cuda_delta_net_step(d_h, d_k_vh, d_v_vh, d_q_vh,
                                             gg, bg, d_out_vh, stream);
                }
            }
        }
        cudaStreamSynchronize(stream);

        // ===== Step 10: Gated normalization =====
        wubu_cuda_silu(N * VALUE_DIM, d_z, d_z_silu, stream);
        wubu_cuda_gated_norm(B, T, SSM_V_HEADS, SSM_D_STATE,
                             d_delta_out, d_ssm_norm, d_z_silu, stream);

        // ===== Step 11: Output projection =====
        wubu_cuda_matmul(cublas_h, d_delta_out, N, VALUE_DIM, d_ssm_out, D_MODEL, d_output, 1.0f, 0.0f);

        cudaStreamSynchronize(stream);
        double t_gpu = now_sec() - t1;

        // Copy results back
        float *output_gpu = (float *)malloc(N * D_MODEL * sizeof(float));
        cudaMemcpy(output_gpu, d_output, N * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost);

        // Also copy state back for comparison
        float *ssm_state_gpu = (float *)malloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
        cudaMemcpy(ssm_state_gpu, d_ssm_state,
                   SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // ========== Compare ==========
        float max_diff = max_abs_diff(output_cpu, output_gpu, N * D_MODEL);
        float max_state_diff = max_abs_diff(ssm_state_cpu, ssm_state_gpu,
                                            SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE);
        printf("\n=== Results ===\n");
        printf("GPU forward pass: %.3f ms\n", t_gpu * 1000.0);
        printf("GPU output[0:8]:");
        for (int i = 0; i < 8; i++) printf(" %+.6f", output_gpu[i]);
        printf("\n");
        printf("CPU output[0:8]:");
        for (int i = 0; i < 8; i++) printf(" %+.6f", output_cpu[i]);
        printf("\n");
        printf("Max diff (GPU vs CPU): %.8f\n", max_diff);
        printf("Max state diff:        %.8f\n", max_state_diff);

        if (max_diff < 1e-3f && max_state_diff < 1e-3f) {
            printf("PASS: GPU/CPU match within tolerance (1e-3)\n");
        } else if (max_diff < 1e-1f) {
            printf("WARN: Moderate divergence — check cuBLAS matmul ordering (transposition)\n");
        } else {
            printf("FAIL: Large divergence — GPU implementation has bugs\n");
        }

        // Cleanup
        free(output_gpu);
        free(ssm_state_gpu);
        free(output_cpu);
        free(ssm_state_cpu);
        free(conv_state_cpu);
        free(x_cpu);

        // Free CPU weights
        free(w_cpu.attn_qkv_weight);
        free(w_cpu.attn_gate_weight);
        free(w_cpu.ssm_beta_weight);
        free(w_cpu.ssm_alpha_weight);
        free(w_cpu.ssm_dt_bias);
        free(w_cpu.ssm_a);
        free(w_cpu.ssm_conv1d_weight);
        free(w_cpu.ssm_norm_weight);
        free(w_cpu.ssm_out_weight);

        // Free GPU
        wubu_cuda_free(d_attn_qkv);   wubu_cuda_free(d_attn_gate);
        wubu_cuda_free(d_ssm_beta);   wubu_cuda_free(d_ssm_alpha);
        wubu_cuda_free(d_ssm_dt_bias); wubu_cuda_free(d_ssm_a);
        wubu_cuda_free(d_ssm_conv1d); wubu_cuda_free(d_ssm_norm);
        wubu_cuda_free(d_ssm_out);
        wubu_cuda_free(d_x);       wubu_cuda_free(d_output);
        wubu_cuda_free(d_ssm_state); wubu_cuda_free(d_conv_state);
        wubu_cuda_free(d_qkv);     wubu_cuda_free(d_z);
        wubu_cuda_free(d_beta);    wubu_cuda_free(d_alpha);
        wubu_cuda_free(d_beta_sig); wubu_cuda_free(d_alpha_bi);
        wubu_cuda_free(d_gate);    wubu_cuda_free(d_conv_input);
        wubu_cuda_free(d_conv_out); wubu_cuda_free(d_q_conv);
        wubu_cuda_free(d_k_conv);  wubu_cuda_free(d_v_conv);
        wubu_cuda_free(d_q_norm);  wubu_cuda_free(d_k_norm);
        wubu_cuda_free(d_delta_out); wubu_cuda_free(d_z_silu);

        wubu_cuda_destroy(cublas_h, stream);
        return 0;

    fail:
        free(w_cpu.attn_qkv_weight);
        free(w_cpu.attn_gate_weight);
        free(w_cpu.ssm_beta_weight);
        free(w_cpu.ssm_alpha_weight);
        free(w_cpu.ssm_dt_bias);
        free(w_cpu.ssm_a);
        free(w_cpu.ssm_conv1d_weight);
        free(w_cpu.ssm_norm_weight);
        free(w_cpu.ssm_out_weight);
        gguf_close(ctx);
        wubu_cuda_destroy(cublas_h, stream);
        return 1;

    cleanup:
        wubu_cuda_destroy(cublas_h, stream);
        return 1;

    } else {
        // ---- GQA layer ----
        gqa_layer_weights w_cpu;
        memset(&w_cpu, 0, sizeof(w_cpu));

        char name[256];
        gguf_tensor_info *t;

        int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;        // 4096
        int q_dim_x2 = q_dim * 2;                       // 8192
        int kv_dim = GQA_KV_HEADS * GQA_HEAD_DIM;       // 512

        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", test_layer);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
        w_cpu.attn_q_weight = (float *)malloc(D_MODEL * q_dim_x2 * sizeof(float));
        gguf_read_tensor_f32(ctx, t, w_cpu.attn_q_weight, D_MODEL * q_dim_x2);
        printf("  attn_q read %d elems\n", D_MODEL * q_dim_x2);

        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", test_layer);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
        w_cpu.attn_k_weight = (float *)malloc(D_MODEL * kv_dim * sizeof(float));
        gguf_read_tensor_f32(ctx, t, w_cpu.attn_k_weight, D_MODEL * kv_dim);

        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", test_layer);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
        w_cpu.attn_v_weight = (float *)malloc(D_MODEL * kv_dim * sizeof(float));
        gguf_read_tensor_f32(ctx, t, w_cpu.attn_v_weight, D_MODEL * kv_dim);

        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", test_layer);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
        w_cpu.attn_output_weight = (float *)malloc(q_dim * D_MODEL * sizeof(float));
        gguf_read_tensor_f32(ctx, t, w_cpu.attn_output_weight, q_dim * D_MODEL);

        snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", test_layer);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
        w_cpu.attn_q_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
        gguf_read_tensor_f32(ctx, t, w_cpu.attn_q_norm_weight, GQA_HEAD_DIM);

        snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", test_layer);
        t = gguf_find_tensor(ctx, name);
        if (!t) { fprintf(stderr, "Missing %s\n", name); goto fail; }
        w_cpu.attn_k_norm_weight = (float *)malloc(GQA_HEAD_DIM * sizeof(float));
        gguf_read_tensor_f32(ctx, t, w_cpu.attn_k_norm_weight, GQA_HEAD_DIM);

        gguf_close(ctx);
        printf("GQA weights loaded (6 tensors)\n");

        // ========== Input data ==========
        const int N = B * T;
        float *x_cpu = (float *)malloc(N * D_MODEL * sizeof(float));
        srand(42);
        for (int i = 0; i < N * D_MODEL; i++) {
            x_cpu[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }

        // ========== CPU forward pass ==========
        float *output_cpu = (float *)calloc(N * D_MODEL, sizeof(float));
        double t0 = now_sec();
        wubu_gqa_forward(x_cpu, B, T, &w_cpu, output_cpu);
        double t_cpu = now_sec() - t0;
        printf("CPU forward pass: %.3f ms\n", t_cpu * 1000.0);
        printf("CPU output[0:8]:");
        for (int i = 0; i < 8; i++) printf(" %+.6f", output_cpu[i]);
        printf("\n");
        printf("CPU output max val: %.6f\n", max_abs_val(output_cpu, N * D_MODEL));

        // ========== GPU forward pass ==========
        printf("Running GQA GPU forward pass...\n");

        // Allocate GPU memory for weights
        float *d_attn_q     = wubu_cuda_alloc(D_MODEL * q_dim_x2 * sizeof(float));
        float *d_attn_k     = wubu_cuda_alloc(D_MODEL * kv_dim * sizeof(float));
        float *d_attn_v     = wubu_cuda_alloc(D_MODEL * kv_dim * sizeof(float));
        float *d_attn_out_w = wubu_cuda_alloc(q_dim * D_MODEL * sizeof(float));
        float *d_q_norm_w   = wubu_cuda_alloc(GQA_HEAD_DIM * sizeof(float));
        float *d_k_norm_w   = wubu_cuda_alloc(GQA_HEAD_DIM * sizeof(float));

        if (!d_attn_q || !d_attn_k || !d_attn_v || !d_attn_out_w || !d_q_norm_w || !d_k_norm_w) {
            fprintf(stderr, "GPU weight allocation failed\n");
            goto cleanup_gqa;
        }

        wubu_cuda_to_device(w_cpu.attn_q_weight, d_attn_q, D_MODEL * q_dim_x2 * sizeof(float), stream);
        wubu_cuda_to_device(w_cpu.attn_k_weight, d_attn_k, D_MODEL * kv_dim * sizeof(float), stream);
        wubu_cuda_to_device(w_cpu.attn_v_weight, d_attn_v, D_MODEL * kv_dim * sizeof(float), stream);
        wubu_cuda_to_device(w_cpu.attn_output_weight, d_attn_out_w, q_dim * D_MODEL * sizeof(float), stream);
        wubu_cuda_to_device(w_cpu.attn_q_norm_weight, d_q_norm_w, GQA_HEAD_DIM * sizeof(float), stream);
        wubu_cuda_to_device(w_cpu.attn_k_norm_weight, d_k_norm_w, GQA_HEAD_DIM * sizeof(float), stream);

        // GPU buffers
        float *d_x        = wubu_cuda_alloc(N * D_MODEL * sizeof(float));
        float *d_output   = wubu_cuda_alloc(N * D_MODEL * sizeof(float));
        float *d_Q_full   = wubu_cuda_alloc(N * q_dim_x2 * sizeof(float));
        float *d_K        = wubu_cuda_alloc(N * kv_dim * sizeof(float));
        float *d_V        = wubu_cuda_alloc(N * kv_dim * sizeof(float));
        float *d_scratch  = wubu_cuda_alloc(N * q_dim * sizeof(float));

        if (!d_x || !d_output || !d_Q_full || !d_K || !d_V || !d_scratch) {
            fprintf(stderr, "GQA GPU buffer allocation failed\n");
            goto cleanup_gqa;
        }

        wubu_cuda_to_device(x_cpu, d_x, N * D_MODEL * sizeof(float), stream);
        cudaStreamSynchronize(stream);
        double t1 = now_sec();

        // ===== Step 1: Q + gate fused projection =====
        wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_q, q_dim_x2, d_Q_full, 1.0f, 0.0f);

        // ===== Step 2: K projection =====
        wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_k, kv_dim, d_K, 1.0f, 0.0f);

        wubu_cuda_matmul(cublas_h, d_x, N, D_MODEL, d_attn_v, kv_dim, d_V, 1.0f, 0.0f);

        // ===== Steps 3-7: Fused GQA attention =====
        wubu_cuda_gqa_forward(cublas_h, stream,
            B, T,
            d_Q_full, d_K, d_V,
            d_q_norm_w, d_k_norm_w,
            d_attn_out_w,
            d_output, d_scratch);

        cudaStreamSynchronize(stream);

        // ===== Copy results back =====
        float *output_gpu = (float *)malloc(N * D_MODEL * sizeof(float));
        cudaMemcpy(output_gpu, d_output, N * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost);

        double t_gpu = now_sec() - t1;

        // ========== Compare ==========
        float max_diff = max_abs_diff(output_cpu, output_gpu, N * D_MODEL);
        printf("\n=== Results ===\n");
        printf("GPU forward pass: %.3f ms\n", t_gpu * 1000.0);
        printf("GPU output[0:8]:");
        for (int i = 0; i < 8; i++) printf(" %+.6f", output_gpu[i]);
        printf("\n");
        printf("CPU output[0:8]:");
        for (int i = 0; i < 8; i++) printf(" %+.6f", output_cpu[i]);
        printf("\n");
        printf("Max diff (GPU vs CPU): %.8f\n", max_diff);

        if (max_diff < 1e-3f) {
            printf("PASS: GPU/CPU match within tolerance (1e-3)\n");
        } else if (max_diff < 1e-1f) {
            printf("WARN: Moderate divergence\n");
        } else {
            printf("FAIL: Large divergence\n");
        }

        // Cleanup
        free(output_gpu);
        free(output_cpu);
        free(x_cpu);

        free(w_cpu.attn_q_weight);
        free(w_cpu.attn_k_weight);
        free(w_cpu.attn_v_weight);
        free(w_cpu.attn_output_weight);
        free(w_cpu.attn_q_norm_weight);
        free(w_cpu.attn_k_norm_weight);

        wubu_cuda_free(d_attn_q);      wubu_cuda_free(d_attn_k);
        wubu_cuda_free(d_attn_v);      wubu_cuda_free(d_attn_out_w);
        wubu_cuda_free(d_q_norm_w);    wubu_cuda_free(d_k_norm_w);
        wubu_cuda_free(d_x);           wubu_cuda_free(d_output);
        wubu_cuda_free(d_Q_full);      wubu_cuda_free(d_K);
        wubu_cuda_free(d_V);           wubu_cuda_free(d_scratch);

        wubu_cuda_destroy(cublas_h, stream);
        return 0;

    cleanup_gqa:
        wubu_cuda_free(d_attn_q);      wubu_cuda_free(d_attn_k);
        wubu_cuda_free(d_attn_v);      wubu_cuda_free(d_attn_out_w);
        wubu_cuda_free(d_q_norm_w);    wubu_cuda_free(d_k_norm_w);
        wubu_cuda_free(d_x);           wubu_cuda_free(d_output);
        wubu_cuda_free(d_Q_full);      wubu_cuda_free(d_K);
        wubu_cuda_free(d_V);           wubu_cuda_free(d_scratch);
        wubu_cuda_destroy(cublas_h, stream);
        return 1;
    }
}
