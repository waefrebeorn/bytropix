/**
 * test_gpu_poincare.c — Verify GPU Poincaré SSM forward produces valid output
 */
#include "bench.h"
#include "wubu_ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const int B = 1, T = 4, N = B * T;
    float R = 0.956f;

    // Load layer 0 weights
    gguf_ctx *ctx = gguf_open(path);
    ssm_layer_weights w;
    memset(&w, 0, sizeof(w));
    int qkv_dim = KEY_DIM * 2 + VALUE_DIM;
    
    #define LOAD_SSM_N(n, f, sz) do { \
        gguf_tensor_info *t = gguf_find_tensor(ctx, "blk.0." n); \
        w.f = malloc(sz * sizeof(float)); \
        gguf_read_tensor_f32(ctx, t, w.f, sz); \
    } while(0)
    
    LOAD_SSM_N("attn_qkv.weight", attn_qkv_weight, D_MODEL * qkv_dim);
    LOAD_SSM_N("attn_gate.weight", attn_gate_weight, D_MODEL * VALUE_DIM);
    LOAD_SSM_N("ssm_beta.weight", ssm_beta_weight, D_MODEL * DT_RANK);
    LOAD_SSM_N("ssm_alpha.weight", ssm_alpha_weight, D_MODEL * DT_RANK);
    LOAD_SSM_N("ssm_dt.bias", ssm_dt_bias, DT_RANK);
    LOAD_SSM_N("ssm_a", ssm_a, DT_RANK);
    LOAD_SSM_N("ssm_conv1d.weight", ssm_conv1d_weight, CONV_KERNEL * CONV_DIM);
    LOAD_SSM_N("ssm_norm.weight", ssm_norm_weight, SSM_D_STATE);
    LOAD_SSM_N("ssm_out.weight", ssm_out_weight, VALUE_DIM * D_MODEL);
    gguf_close(ctx);

    // Random input
    float *x = (float *)malloc(N * D_MODEL * sizeof(float));
    srand(42);
    for (int i = 0; i < N * D_MODEL; i++)
        x[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    // GPU init
    cublasHandle_t cublas_h;
    cudaStream_t stream;
    wubu_cuda_init(&cublas_h, &stream);

    // Allocate GPU buffers
    float *d_x = wubu_cuda_alloc(N * D_MODEL * sizeof(float));
    float *d_out = wubu_cuda_alloc(N * D_MODEL * sizeof(float));
    
    float *d_ssm_s = wubu_cuda_alloc(SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float));
    float *d_conv_s = wubu_cuda_alloc((CONV_KERNEL - 1) * CONV_DIM * sizeof(float));
    
    // Scratch buffers
    float *d_qkv = wubu_cuda_alloc(N * qkv_dim * sizeof(float));
    float *d_z = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));
    float *d_beta = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    float *d_alpha = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    float *d_bs = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    float *d_ab = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    float *d_gate = wubu_cuda_alloc(N * DT_RANK * sizeof(float));
    float *d_ci = wubu_cuda_alloc(B * (T + CONV_KERNEL - 1) * CONV_DIM * sizeof(float));
    float *d_co = wubu_cuda_alloc(N * CONV_DIM * sizeof(float));
    float *d_q = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
    float *d_k = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
    float *d_v = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));
    float *d_qn = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
    float *d_kn = wubu_cuda_alloc(N * KEY_DIM * sizeof(float));
    float *d_dd = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));
    float *d_zs = wubu_cuda_alloc(N * VALUE_DIM * sizeof(float));

    // Upload weights + input
    gpu_ssm_weights gw;
    ctx = gguf_open(path);
    gpu_load_ssm_layer(ctx, 0, &gw, stream);
    gguf_close(ctx);
    wubu_cuda_to_device(x, d_x, N * D_MODEL * sizeof(float), stream);

    // === Euclidean GPU forward ===
    cudaMemsetAsync(d_ssm_s, 0, SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float), stream);
    cudaMemsetAsync(d_conv_s, 0, (CONV_KERNEL - 1) * CONV_DIM * sizeof(float), stream);
    gpu_ssm_forward(cublas_h, stream, d_x, B, T,
        gw.d_attn_qkv, gw.d_attn_gate, gw.d_ssm_beta, gw.d_ssm_alpha,
        gw.d_ssm_dt_bias, gw.d_ssm_a, gw.d_ssm_conv1d, gw.d_ssm_norm, gw.d_ssm_out,
        d_ssm_s, d_conv_s, d_out,
        d_qkv, d_z, d_beta, d_alpha, d_bs, d_ab, d_gate,
        d_ci, d_co, d_q, d_k, d_v, d_qn, d_kn, d_dd, d_zs);
    
    float *euc = (float *)malloc(N * D_MODEL * sizeof(float));
    cudaMemcpy(euc, d_out, N * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost);

    // === Poincaré GPU forward ===
    cudaMemsetAsync(d_ssm_s, 0, SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE * sizeof(float), stream);
    cudaMemsetAsync(d_conv_s, 0, (CONV_KERNEL - 1) * CONV_DIM * sizeof(float), stream);
    gpu_poincare_ssm_forward(cublas_h, stream, d_x, B, T,
        gw.d_attn_qkv, gw.d_attn_gate, gw.d_ssm_beta, gw.d_ssm_alpha,
        gw.d_ssm_dt_bias, gw.d_ssm_a, gw.d_ssm_conv1d, gw.d_ssm_norm, gw.d_ssm_out,
        d_ssm_s, d_conv_s, d_out,
        d_qkv, d_z, d_beta, d_alpha, d_bs, d_ab, d_gate,
        d_ci, d_co, d_q, d_k, d_v, d_qn, d_kn, d_dd, d_zs,
        R);

    cudaStreamSynchronize(stream);
    float *poi = (float *)malloc(N * D_MODEL * sizeof(float));
    cudaMemcpy(poi, d_out, N * D_MODEL * sizeof(float), cudaMemcpyDeviceToHost);

    // Stats
    float e_min = euc[0], e_max = euc[0];
    float p_min = poi[0], p_max = poi[0];
    int e_nan = 0, p_nan = 0, e_inf = 0, p_inf = 0;
    for (int i = 0; i < N * D_MODEL; i++) {
        if (isnan(euc[i])) e_nan++;
        if (isinf(euc[i])) e_inf++;
        if (isnan(poi[i])) p_nan++;
        if (isinf(poi[i])) p_inf++;
        if (euc[i] < e_min) e_min = euc[i];
        if (euc[i] > e_max) e_max = euc[i];
        if (poi[i] < p_min) p_min = poi[i];
        if (poi[i] > p_max) p_max = poi[i];
    }
    printf("Euclidean GPU: range=[%.4f, %.4f] NaN=%d Inf=%d\n", e_min, e_max, e_nan, e_inf);
    printf("Poincaré GPU:  range=[%.4f, %.4f] NaN=%d Inf=%d\n", p_min, p_max, p_nan, p_inf);
    
    if (p_nan == 0 && p_inf == 0 && fabs(p_min) > 1e-10f)
        printf("PASS: Poincaré GPU forward produces valid non-zero output\n");
    else
        printf("FAIL: Poincaré GPU forward has issues\n");
    
    return 0;
}
