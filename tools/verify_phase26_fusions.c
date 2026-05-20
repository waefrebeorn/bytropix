/**
 * Verify Phase 26 fused kernels vs old cuBLAS+element-wise path.
 * Tests: ssm_beta_alpha_fused_decode + ssm_conv_silu_split_decode
 * Only activates for N=1 decode — these kernels don't handle C>1.
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

static double max_abs_diff(const float *a, const float *b, int n) {
    double md = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > md) md = d;
    }
    return md;
}

static double cos_sim(const float *a, const float *b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    double denom = sqrt(na) * sqrt(nb);
    return (denom > 1e-30) ? dot / denom : 1.0;
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *gguf_path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    int layer = argc > 2 ? atoi(argv[2]) : 0;
    printf("Phase 26 Fused Kernel Verification — Layer %d\n", layer);
    printf("Model: %s\n\n", gguf_path);

    // Load layer weights as F32 (dequantized reference)
    gguf_ctx *ctx = gguf_open(gguf_path);
    if (!ctx) { fprintf(stderr, "FAIL: gguf_open\n"); return 1; }

    // Tensor names
    char name_qkv[64], name_gate[64], name_beta[64], name_alpha[64];
    char name_dt[64], name_a[64], name_conv[64], name_norm[64], name_out[64];
    snprintf(name_qkv, 64, "blk.%d.attn_qkv.weight", layer);
    snprintf(name_gate, 64, "blk.%d.attn_gate.weight", layer);
    snprintf(name_beta, 64, "blk.%d.ssm_beta.weight", layer);

    // Check actual GGUF tensor dims
    gguf_tensor_info *t_beta = gguf_find_tensor(ctx, name_beta);
    if (t_beta) {
        printf("ssm_beta.weight GGUF dims: ");
        for (int d = 0; d < t_beta->n_dims; d++) printf("%ld ", (long)t_beta->dims[d]);
        printf("\n");
    }
    gguf_tensor_info *t_alpha = gguf_find_tensor(ctx, name_alpha);
    if (t_alpha) {
        printf("ssm_alpha.weight GGUF dims: ");
        for (int d = 0; d < t_alpha->n_dims; d++) printf("%ld ", (long)t_alpha->dims[d]);
        printf("\n");
    }
    snprintf(name_alpha, 64, "blk.%d.ssm_alpha.weight", layer);
    snprintf(name_dt, 64, "blk.%d.ssm_dt.bias", layer);
    snprintf(name_a, 64, "blk.%d.ssm_a", layer);
    snprintf(name_conv, 64, "blk.%d.ssm_conv1d.weight", layer);
    snprintf(name_norm, 64, "blk.%d.ssm_norm.weight", layer);
    snprintf(name_out, 64, "blk.%d.ssm_out.weight", layer);

    float *w_qkv, *w_gate, *w_beta, *w_alpha;
    float *w_dt, *w_a, *w_conv, *w_norm, *w_out;

    #define LOAD_T(name, buf, n) do { \
        gguf_tensor_info *t_ = gguf_find_tensor(ctx, name); \
        if (!t_) { fprintf(stderr, "MISSING %s\n", name); return 1; } \
        buf = (float*)malloc((n) * sizeof(float)); \
        gguf_read_tensor_f32(ctx, t_, buf, n); \
    } while(0)

    LOAD_T(name_qkv, w_qkv, D_MODEL * CONV_DIM);
    LOAD_T(name_gate, w_gate, D_MODEL * VALUE_DIM);
    LOAD_T(name_beta, w_beta, D_MODEL * DT_RANK);
    LOAD_T(name_alpha, w_alpha, D_MODEL * DT_RANK);
    LOAD_T(name_dt, w_dt, DT_RANK);
    LOAD_T(name_a, w_a, DT_RANK);
    LOAD_T(name_conv, w_conv, CONV_KERNEL * CONV_DIM);
    LOAD_T(name_norm, w_norm, SSM_D_STATE);
    LOAD_T(name_out, w_out, VALUE_DIM * D_MODEL);
    gguf_close(ctx);
    printf("Loaded weights OK.\n");

    // Init CUDA
    cublasHandle_t handle;
    cudaStream_t stream;
    wubu_cuda_init(&handle, &stream);

    const int N = 1;  // N=1 decode only — fused kernels don't support C>1
    const int qdim = CONV_DIM, vdim = VALUE_DIM, kdim = KEY_DIM, dr = DT_RANK;
    const int k = CONV_KERNEL, k1 = k - 1;

    // Random input
    float *x = (float*)malloc(N * D_MODEL * sizeof(float));
    srand(42);
    for (int i = 0; i < N * D_MODEL; i++)
        x[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;

    // === Allocate GPU memory ===
    float *d_w_qkv   = wubu_cuda_alloc((size_t)D_MODEL * qdim * sizeof(float));
    float *d_w_gate  = wubu_cuda_alloc((size_t)D_MODEL * vdim * sizeof(float));
    float *d_w_beta  = wubu_cuda_alloc((size_t)D_MODEL * dr * sizeof(float));
    float *d_w_alpha = wubu_cuda_alloc((size_t)D_MODEL * dr * sizeof(float));
    float *d_dt      = wubu_cuda_alloc((size_t)dr * sizeof(float));
    float *d_a       = wubu_cuda_alloc((size_t)dr * sizeof(float));
    float *d_conv    = wubu_cuda_alloc((size_t)k * qdim * sizeof(float));
    float *d_norm    = wubu_cuda_alloc((size_t)SSM_D_STATE * sizeof(float));
    float *d_out     = wubu_cuda_alloc((size_t)vdim * D_MODEL * sizeof(float));
    float *d_x       = wubu_cuda_alloc((size_t)N * D_MODEL * sizeof(float));
    float *d_conv_state = wubu_cuda_alloc((size_t)k1 * qdim * sizeof(float));
    float *d_scratch = NULL; // will allocate as needed

    size_t scratch_needed = (size_t)4 * qdim + (size_t)4 * vdim + (size_t)4 * dr + (size_t)N * D_MODEL;
    d_scratch = wubu_cuda_alloc(scratch_needed * sizeof(float));

    // Transfer weights
    wubu_cuda_to_device(w_qkv,   d_w_qkv,   (size_t)D_MODEL * qdim * sizeof(float), stream);
    wubu_cuda_to_device(w_gate,  d_w_gate,  (size_t)D_MODEL * vdim * sizeof(float), stream);
    wubu_cuda_to_device(w_beta,  d_w_beta,  (size_t)D_MODEL * dr * sizeof(float), stream);
    wubu_cuda_to_device(w_alpha, d_w_alpha, (size_t)D_MODEL * dr * sizeof(float), stream);
    wubu_cuda_to_device(w_dt,    d_dt,      (size_t)dr * sizeof(float), stream);
    wubu_cuda_to_device(w_a,     d_a,       (size_t)dr * sizeof(float), stream);
    wubu_cuda_to_device(w_conv,  d_conv,    (size_t)k * qdim * sizeof(float), stream);
    wubu_cuda_to_device(w_norm,  d_norm,    (size_t)SSM_D_STATE * sizeof(float), stream);
    wubu_cuda_to_device(w_out,   d_out,     (size_t)vdim * D_MODEL * sizeof(float), stream);
    wubu_cuda_to_device(x,       d_x,       (size_t)N * D_MODEL * sizeof(float), stream);
    cudaMemset(d_conv_state, 0, (size_t)k1 * qdim * sizeof(float));
    cudaStreamSynchronize(stream);

    printf("\n=== Verification 1: ssm_beta_alpha_fused_decode ===\n");
    {
        // --- OLD PATH: cuBLAS matmul + element-wise ---
        float *d_beta_raw  = d_scratch;
        float *d_alpha_raw = d_scratch + (size_t)N * dr;
        float *d_alpha_bias= d_scratch + (size_t)2 * N * dr;
        float *d_alpha_sp  = d_scratch + (size_t)3 * N * dr;
        float *d_beta_sig_old = d_scratch + (size_t)4 * N * dr;
        float *d_gate_old     = d_scratch + (size_t)5 * N * dr;

        double t0 = now_sec();
        wubu_cuda_matmul(handle, d_x, N, D_MODEL, d_w_beta, dr, d_beta_raw, 1.0f, 0.0f);
        wubu_cuda_matmul(handle, d_x, N, D_MODEL, d_w_alpha, dr, d_alpha_raw, 1.0f, 0.0f);
        wubu_cuda_sigmoid((size_t)N * dr, d_beta_raw, d_beta_sig_old, stream);
        wubu_cuda_add_bias(N, dr, d_alpha_raw, d_dt, d_alpha_bias, stream);
        wubu_cuda_softplus((size_t)N * dr, d_alpha_bias, d_alpha_sp, stream);
        wubu_cuda_mul_by_scalar(N, dr, d_alpha_sp, d_a, d_gate_old, stream);
        cudaStreamSynchronize(stream);
        double t_old = now_sec() - t0;

        float *beta_old = (float*)malloc((size_t)N * dr * sizeof(float));
        float *gate_old = (float*)malloc((size_t)N * dr * sizeof(float));
        cudaMemcpy(beta_old, d_beta_sig_old, (size_t)N * dr * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gate_old, d_gate_old, (size_t)N * dr * sizeof(float), cudaMemcpyDeviceToHost);

        // --- NEW PATH: ssm_beta_alpha_fused_decode kernel ---
        float *d_beta_sig_new = d_scratch + (size_t)6 * N * dr;
        float *d_gate_new     = d_scratch + (size_t)7 * N * dr;

        double t1 = now_sec();
        ssm_beta_alpha_fused_decode_wrapper(stream,
            d_x, d_w_beta, d_w_alpha, d_dt, d_a,
            d_beta_sig_new, d_gate_new, dr);
        cudaStreamSynchronize(stream);
        double t_new = now_sec() - t1;

        float *beta_new = (float*)malloc((size_t)N * dr * sizeof(float));
        float *gate_new = (float*)malloc((size_t)N * dr * sizeof(float));
        cudaMemcpy(beta_new, d_beta_sig_new, (size_t)N * dr * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(gate_new, d_gate_new, (size_t)N * dr * sizeof(float), cudaMemcpyDeviceToHost);

        printf("beta  cos-sim: %.8f  max|diff|: %.10f\n",
               cos_sim(beta_old, beta_new, N * dr),
               max_abs_diff(beta_old, beta_new, N * dr));
        printf("gate  cos-sim: %.8f  max|diff|: %.10f\n",
               cos_sim(gate_old, gate_new, N * dr),
               max_abs_diff(gate_old, gate_new, N * dr));
        printf("Old path: %.3f ms  New path: %.3f ms\n", t_old * 1000, t_new * 1000);

        free(beta_old); free(gate_old);
        free(beta_new); free(gate_new);
    }

    printf("\n=== Verification 2: ssm_conv_silu_split_decode ===\n");
    {
        // Run qkv + gate matmuls first (shared by both old/new)
        float *d_qkv = wubu_cuda_alloc((size_t)N * qdim * sizeof(float));
        float *d_z   = wubu_cuda_alloc((size_t)N * vdim * sizeof(float));

        wubu_cuda_matmul(handle, d_x, N, D_MODEL, d_w_qkv, qdim, d_qkv, 1.0f, 0.0f);
        wubu_cuda_matmul(handle, d_x, N, D_MODEL, d_w_gate, vdim, d_z, 1.0f, 0.0f);
        cudaStreamSynchronize(stream);

        // --- OLD PATH: conv1d + silu + split_qkv + conv_state update ---
        float *d_conv_input = wubu_cuda_alloc((size_t)(k1 + N) * qdim * sizeof(float));
        float *d_conv_output = wubu_cuda_alloc((size_t)N * qdim * sizeof(float));
        float *d_q_old = wubu_cuda_alloc((size_t)N * kdim * sizeof(float));
        float *d_k_old = wubu_cuda_alloc((size_t)N * kdim * sizeof(float));
        float *d_v_old = wubu_cuda_alloc((size_t)N * vdim * sizeof(float));
        float *d_cs_new = wubu_cuda_alloc((size_t)k1 * qdim * sizeof(float));

        double t0 = now_sec();
        // Build conv input: [conv_state | qkv_out]
        cudaMemcpyAsync(d_conv_input, d_conv_state,
                        (size_t)k1 * qdim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_conv_input + (size_t)k1 * qdim, d_qkv,
                        (size_t)N * qdim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        // Conv1d
        wubu_cuda_conv1d(1, N, qdim, k, d_conv_input, d_conv, d_conv_output, stream);
        // SiLU
        wubu_cuda_silu((size_t)N * qdim, d_conv_output, d_conv_output, stream);
        // Split QKV
        wubu_cuda_split_qkv(N, kdim, vdim, d_conv_output, d_q_old, d_k_old, d_v_old, stream);
        // Update conv_state
        cudaMemcpyAsync(d_cs_new, d_conv_input + (size_t)N * qdim,
                        (size_t)k1 * qdim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
        double t_old = now_sec() - t0;

        float *q_old = (float*)malloc((size_t)N * kdim * sizeof(float));
        float *k_old = (float*)malloc((size_t)N * kdim * sizeof(float));
        float *v_old = (float*)malloc((size_t)N * vdim * sizeof(float));
        cudaMemcpy(q_old, d_q_old, (size_t)N * kdim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(k_old, d_k_old, (size_t)N * kdim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(v_old, d_v_old, (size_t)N * vdim * sizeof(float), cudaMemcpyDeviceToHost);

        // --- NEW PATH: ssm_conv_silu_split_decode kernel ---
        float *d_q_new = wubu_cuda_alloc((size_t)N * kdim * sizeof(float));
        float *d_k_new = wubu_cuda_alloc((size_t)N * kdim * sizeof(float));
        float *d_v_new = wubu_cuda_alloc((size_t)N * vdim * sizeof(float));
        float *d_cs_new2 = wubu_cuda_alloc((size_t)k1 * qdim * sizeof(float));

        // Reset conv_state to same initial state
        cudaMemset(d_conv_state, 0, (size_t)k1 * qdim * sizeof(float));
        cudaStreamSynchronize(stream);

        double t1 = now_sec();
        ssm_conv_silu_split_decode_wrapper(stream,
            d_conv_state, d_qkv, d_conv,
            d_q_new, d_k_new, d_v_new,
            d_cs_new2);
        cudaStreamSynchronize(stream);
        double t_new = now_sec() - t1;

        float *q_new = (float*)malloc((size_t)N * kdim * sizeof(float));
        float *k_new = (float*)malloc((size_t)N * kdim * sizeof(float));
        float *v_new = (float*)malloc((size_t)N * vdim * sizeof(float));
        cudaMemcpy(q_new, d_q_new, (size_t)N * kdim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(k_new, d_k_new, (size_t)N * kdim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(v_new, d_v_new, (size_t)N * vdim * sizeof(float), cudaMemcpyDeviceToHost);

        printf("Q     cos-sim: %.8f  max|diff|: %.10f\n",
               cos_sim(q_old, q_new, N * kdim),
               max_abs_diff(q_old, q_new, N * kdim));
        printf("K     cos-sim: %.8f  max|diff|: %.10f\n",
               cos_sim(k_old, k_new, N * kdim),
               max_abs_diff(k_old, k_new, N * kdim));
        printf("V     cos-sim: %.8f  max|diff|: %.10f\n",
               cos_sim(v_old, v_new, N * vdim),
               max_abs_diff(v_old, v_new, N * vdim));
        printf("Old path: %.3f ms  New path: %.3f ms\n", t_old * 1000, t_new * 1000);

        // Also print first few values for debugging
        printf("\nQ first 5 (old): ");
        for (int i = 0; i < 5; i++) printf("%.6f ", q_old[i]);
        printf("\nQ first 5 (new): ");
        for (int i = 0; i < 5; i++) printf("%.6f ", q_new[i]);
        printf("\n");

        // Debug: dump weight layout mismatch
        printf("\n=== WEIGHT LAYOUT DEBUG ===\n");
        printf("w_beta[0..4]: ");
        for (int i = 0; i < 5; i++) printf("%.6f ", w_beta[i]);
        printf("\nw_beta[2048..2052]: ");
        for (int i = 2048; i < 2053; i++) printf("%.6f ", w_beta[i]);
        printf("\nw_beta[4096..4100]: ");
        for (int i = 4096; i < 4101; i++) printf("%.6f ", w_beta[i]);
        
        // For unit vector x[0]=1, x[1..]=0, what should beta be?
        // Row-major: beta[idx] = w_beta[0*32 + idx] for x[0]=1
        // Col-major (cuBLAS transpose): beta[idx] = w_beta[idx*2048 + 0]
        printf("\n\nRow-major beta (x[0]=1): ");
        for (int i = 0; i < 5; i++) printf("%.6f ", w_beta[0*32 + i]);
        printf("\nCol-major beta (x[0]=1): ");
        for (int i = 0; i < 5; i++) printf("%.6f ", w_beta[i*2048 + 0]);

        free(q_old); free(k_old); free(v_old);
        free(q_new); free(k_new); free(v_new);
        wubu_cuda_free(d_q_new); wubu_cuda_free(d_k_new); wubu_cuda_free(d_v_new);
        wubu_cuda_free(d_cs_new2);
        wubu_cuda_free(d_conv_input); wubu_cuda_free(d_conv_output);
        wubu_cuda_free(d_q_old); wubu_cuda_free(d_k_old); wubu_cuda_free(d_v_old);
        wubu_cuda_free(d_cs_new);
        wubu_cuda_free(d_qkv); wubu_cuda_free(d_z);
    }

    printf("\n=== Verification Complete ===\n");
    if (argc > 1) printf("Will verify layer %d\n", layer);

    // Cleanup
    free(x);
    free(w_qkv); free(w_gate); free(w_beta); free(w_alpha);
    free(w_dt); free(w_a); free(w_conv); free(w_norm); free(w_out);

    #define GPU_FREE(p) do { if(p) wubu_cuda_free(p); } while(0)
    GPU_FREE(d_w_qkv); GPU_FREE(d_w_gate); GPU_FREE(d_w_beta); GPU_FREE(d_w_alpha);
    GPU_FREE(d_dt); GPU_FREE(d_a); GPU_FREE(d_conv); GPU_FREE(d_norm); GPU_FREE(d_out);
    GPU_FREE(d_x); GPU_FREE(d_conv_state); GPU_FREE(d_scratch);

    wubu_cuda_destroy(handle, stream);
    return 0;
}
