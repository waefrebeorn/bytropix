#ifndef WUBU_BENCH_H
#define WUBU_BENCH_H

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

#ifdef __cplusplus
extern "C" {
#endif

// ================================================================
// Timing helper
// ================================================================
static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ================================================================
// Comparison helpers
// ================================================================
static inline float max_abs_diff(const float *a, const float *b, int n) {
    float md = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > md) md = d;
    }
    return md;
}

static inline float max_abs_val(const float *a, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = fabsf(a[i]);
        if (v > m) m = v;
    }
    return m;
}

// ================================================================
// GPU SSM forward pass (all steps, one layer on GPU)
// Same interface as CPU wubu_ssm_forward but runs on GPU.
// All d_* arrays must be pre-allocated GPU memory.
// ================================================================
void gpu_ssm_forward(cublasHandle_t cublas_h, cudaStream_t stream,
                     const float *d_x, int B, int T,
                     // GPU weight pointers
                     const float *d_attn_qkv,
                     const float *d_attn_gate,
                     const float *d_ssm_beta,
                     const float *d_ssm_alpha,
                     const float *d_ssm_dt_bias,
                     const float *d_ssm_a,
                     const float *d_ssm_conv1d,
                     const float *d_ssm_norm,
                     const float *d_ssm_out,
                     // Mutable state (GPU)
                     float *d_ssm_state,
                     float *d_conv_state,
                     // Output (GPU)
                     float *d_output,
                     // Scratch buffers (GPU, pre-allocated by caller)
                     float *d_qkv,
                     float *d_z,
                     float *d_beta,
                     float *d_alpha,
                     float *d_beta_sig,
                     float *d_alpha_bi,
                     float *d_gate,
                     float *d_conv_input,
                     float *d_conv_out,
                     float *d_q_conv,
                     float *d_k_conv,
                     float *d_v_conv,
                     float *d_q_norm,
                     float *d_k_norm,
                     float *d_delta_out,
                     float *d_z_silu);

// ================================================================
// GPU GQA forward pass
// ================================================================
void gpu_gqa_forward(cublasHandle_t cublas_h, cudaStream_t stream,
                     const float *d_x, int B, int T,
                     const float *d_attn_q,
                     const float *d_attn_k,
                     const float *d_attn_v,
                     const float *d_attn_out_w,
                     const float *d_q_norm_w,
                     const float *d_k_norm_w,
                     // Output (GPU)
                     float *d_output,
                     // Scratch (pre-allocated)
                     float *d_Q_full,
                     float *d_K,
                     float *d_V,
                     float *d_scratch);

// ================================================================
// Load SSM layer weights from GGUF (f32), allocate + upload to GPU
// ================================================================
typedef struct {
    float *d_attn_qkv;
    float *d_attn_gate;
    float *d_ssm_beta;
    float *d_ssm_alpha;
    float *d_ssm_dt_bias;
    float *d_ssm_a;
    float *d_ssm_conv1d;
    float *d_ssm_norm;
    float *d_ssm_out;
} gpu_ssm_weights;

int gpu_load_ssm_layer(gguf_ctx *ctx, int layer_idx,
                       gpu_ssm_weights *w, cudaStream_t stream);

void gpu_free_ssm_weights(gpu_ssm_weights *w);

// ================================================================
// Load GQA layer weights from GGUF (f32), allocate + upload to GPU
// ================================================================
typedef struct {
    float *d_attn_q;
    float *d_attn_k;
    float *d_attn_v;
    float *d_attn_out_w;
    float *d_q_norm_w;
    float *d_k_norm_w;
} gpu_gqa_weights;

int gpu_load_gqa_layer(gguf_ctx *ctx, int layer_idx,
                       gpu_gqa_weights *w, cudaStream_t stream);

void gpu_free_gqa_weights(gpu_gqa_weights *w);

#ifdef __cplusplus
}
#endif

#endif // WUBU_BENCH_H
