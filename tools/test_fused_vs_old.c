/**
 * Compare: Fused SSM Forward (parallel scan) vs step-by-step GPU (old delta_net_step)
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

static double mdf(const float *a, const float *b, int n) {
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
    float *d_h = wubu_cuda_alloc(n_state * sizeof(float));
    float *d_cs = wubu_cuda_alloc(B * 3 * CONV_DIM * sizeof(float));

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
    cudaMemset(d_h, 0, n_state * sizeof(float));
    cudaMemset(d_cs, 0, B * 3 * CONV_DIM * sizeof(float));
    cudaStreamSynchronize(stream);

    // --- Step-by-step GPU (old method) ---
    const int qdim = CONV_DIM, vdim = VALUE_DIM, kdim = KEY_DIM, dr = DT_RANK;
    float *d_qkv = wubu_cuda_alloc(N * qdim * sizeof(float));
    float *d_z = wubu_cuda_alloc(N * vdim * sizeof(float));
    float *d_br = wubu_cuda_alloc(N * dr * sizeof(float));
    float *d_ar = wubu_cuda_alloc(N * dr * sizeof(float));
    float *d_bs = wubu_cuda_alloc(N * dr * sizeof(float));
    float *d_g = wubu_cuda_alloc(N * dr * sizeof(float));
    float *d_ab = wubu_cuda_alloc(N * dr * sizeof(float));
    float *d_ci = wubu_cuda_alloc(B * (T+3) * qdim * sizeof(float));
    float *d_co = wubu_cuda_alloc(N * qdim * sizeof(float));
    float *d_qc = wubu_cuda_alloc(N * kdim * sizeof(float));
    float *d_kc = wubu_cuda_alloc(N * kdim * sizeof(float));
    float *d_vc = wubu_cuda_alloc(N * vdim * sizeof(float));
    float *d_qn = wubu_cuda_alloc(N * kdim * sizeof(float));
    float *d_kn = wubu_cuda_alloc(N * kdim * sizeof(float));
    float *d_do = wubu_cuda_alloc(N * vdim * sizeof(float));
    float *d_zs = wubu_cuda_alloc(N * vdim * sizeof(float));
    float *d_o_old = wubu_cuda_alloc(N * D_MODEL * sizeof(float));

    wubu_cuda_matmul(handle, d_x, N, D_MODEL, d_w_qkv, qdim, d_qkv, 1, 0);
    wubu_cuda_matmul(handle, d_x, N, D_MODEL, d_w_gate, vdim, d_z, 1, 0);
    wubu_cuda_matmul(handle, d_x, N, D_MODEL, d_w_beta, dr, d_br, 1, 0);
    wubu_cuda_matmul(handle, d_x, N, D_MODEL, d_w_alpha, dr, d_ar, 1, 0);
    wubu_cuda_sigmoid(N * dr, d_br, d_bs, stream);
    wubu_cuda_add_bias(N, dr, d_ar, d_dt, d_ab, stream);
    wubu_cuda_softplus(N * dr, d_ab, d_g, stream);
    wubu_cuda_mul_by_scalar(N, dr, d_g, d_a, d_g, stream);
    cudaMemset(d_ci, 0, B * (T+3) * qdim * sizeof(float));
    for (int b = 0; b < B; b++) {
        cudaMemcpyAsync(d_ci + b*(T+3)*qdim + 3*qdim,
                       d_qkv + b*T*qdim, T*qdim*sizeof(float),
                       cudaMemcpyDeviceToDevice, stream);
    }
    wubu_cuda_conv1d(B, T, qdim, CONV_KERNEL, d_ci, d_conv, d_co, stream);
    wubu_cuda_silu(N * qdim, d_co, d_co, stream);
    wubu_cuda_split_qkv(N, kdim, vdim, d_co, d_qc, d_kc, d_vc, stream);
    wubu_cuda_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, d_qc, 1e-12f, d_qn, stream);
    wubu_cuda_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, d_kc, 1e-12f, d_kn, stream);
    int rf = SSM_V_HEADS / SSM_K_HEADS;
    cudaMemset(d_h, 0, n_state * sizeof(float));
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int s = b * T + t;
            float beta_h[32], gate_h[32];
            cudaMemcpyAsync(beta_h, d_bs + s*32, 32*sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(gate_h, d_g + s*32, 32*sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            for (int vh = 0; vh < SSM_V_HEADS; vh++) {
                int kh = vh / rf;
                float *dq = d_qn + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                float *dk = d_kn + (s * SSM_K_HEADS + kh) * SSM_D_STATE;
                float *dv = d_vc + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                float *hh = d_h + (vh * SSM_D_STATE * SSM_D_STATE);
                float *do_ = d_do + (s * SSM_V_HEADS + vh) * SSM_D_STATE;
                wubu_cuda_delta_net_step(hh, dk, dv, dq, gate_h[kh], beta_h[kh], do_, stream);
            }
        }
    }
    cudaStreamSynchronize(stream);
    wubu_cuda_silu(N * vdim, d_z, d_zs, stream);
    wubu_cuda_gated_norm(B, T, SSM_V_HEADS, SSM_D_STATE, d_do, d_norm, d_zs, stream);
    wubu_cuda_matmul(handle, d_do, N, vdim, d_out, D_MODEL, d_o_old, 1, 0);
    cudaStreamSynchronize(stream);

    float *old_h = (float*)malloc(n_state * sizeof(float));
    float *old_o = (float*)malloc(N * D_MODEL * sizeof(float));
    cudaMemcpy(old_o, d_o_old, N * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(old_h, d_h, n_state * sizeof(float), cudaMemcpyDeviceToHost);

    // --- Fused method ---
    size_t sz = wubu_cuda_ssm_forward_query_scratch(B, T);
    float *d_scratch = wubu_cuda_alloc(sz);
    float *d_o_new = wubu_cuda_alloc(N * D_MODEL * sizeof(float));
    cudaMemset(d_h, 0, n_state * sizeof(float));
    cudaMemset(d_cs, 0, B * 3 * CONV_DIM * sizeof(float));
    cudaStreamSynchronize(stream);

    wubu_cuda_ssm_forward(handle, stream, B, T, d_x,
        d_w_qkv, d_w_gate, d_w_beta, d_w_alpha, d_dt, d_a,
        d_conv, d_norm, d_out, d_h, d_cs, d_o_new, d_scratch);
    cudaStreamSynchronize(stream);

    float *new_h = (float*)malloc(n_state * sizeof(float));
    float *new_o = (float*)malloc(N * D_MODEL * sizeof(float));
    cudaMemcpy(new_o, d_o_new, N * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(new_h, d_h, n_state * sizeof(float), cudaMemcpyDeviceToHost);

    printf("=== Fused vs Step-by-Step GPU ===\n");
    printf("Output max diff:  %.10f\n", mdf(old_o, new_o, N * D_MODEL));
    printf("State  max diff:  %.10f\n", mdf(old_h, new_h, n_state));

    // Cleanup
    free(x); free(old_o); free(new_o); free(old_h); free(new_h);
    free(w_qkv); free(w_gate); free(w_beta); free(w_alpha);
    free(w_dt); free(w_a); free(w_conv); free(w_norm); free(w_out);

    #define GPU_FREE(p) do { if(p) wubu_cuda_free(p); } while(0)
    GPU_FREE(d_w_qkv); GPU_FREE(d_w_gate); GPU_FREE(d_w_beta); GPU_FREE(d_w_alpha);
    GPU_FREE(d_dt); GPU_FREE(d_a); GPU_FREE(d_conv); GPU_FREE(d_norm); GPU_FREE(d_out);
    GPU_FREE(d_x); GPU_FREE(d_h); GPU_FREE(d_cs);
    GPU_FREE(d_qkv); GPU_FREE(d_z); GPU_FREE(d_br); GPU_FREE(d_ar);
    GPU_FREE(d_bs); GPU_FREE(d_g); GPU_FREE(d_ab);
    GPU_FREE(d_ci); GPU_FREE(d_co); GPU_FREE(d_qc); GPU_FREE(d_kc); GPU_FREE(d_vc);
    GPU_FREE(d_qn); GPU_FREE(d_kn); GPU_FREE(d_do); GPU_FREE(d_zs);
    GPU_FREE(d_o_old); GPU_FREE(d_scratch); GPU_FREE(d_o_new);

    wubu_cuda_destroy(handle, stream);
    printf("PASS\n");
    return 0;
}
