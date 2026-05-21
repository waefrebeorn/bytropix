#ifndef CUDA_VISION_H
#define CUDA_VISION_H

#include "wubu_vision.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// GPU weight bundle for one vision layer
typedef struct {
    float *d_qkv_w, *d_qkv_b;     // [V_HIDDEN, 3*V_HIDDEN]
    float *d_out_w, *d_out_b;      // [V_HIDDEN, V_HIDDEN]
    float *d_ffn_up_w, *d_ffn_up_b; // [V_HIDDEN, V_INTERMEDIATE]
    float *d_ffn_dn_w, *d_ffn_dn_b; // [V_INTERMEDIATE, V_HIDDEN]
} gpu_vision_weights_t;

// GPU kernel declarations (defined in cuda_vision.cu)
__global__ void attention_kernel(const float *q, const float *k, const float *v,
                                  float *out, int n, int d, float scale);
__global__ void layernorm_kernel(float *x, const float *w, const float *b,
                                  int n, int d, float eps);
__global__ void gelu_kernel(float *x, int n);
__global__ void add_kernel(const float *x, float *y, int n);

// Run one ViT layer on GPU (all linear via cuBLAS, attention via custom kernel)
// d_x: [n, V_HIDDEN] — input (attention/FFN source)
// d_residual: [n, V_HIDDEN] — original pre-LN state for residual (may equal d_x for pre-LN convention)
// If d_residual is NULL, uses d_x as residual (maintains backward compat for single-buffer use)
// scratch: pre-allocated [n * (3*V_HIDDEN + V_HIDDEN*3)] temp buffer
bool gpu_vision_layer_forward(cublasHandle_t cublas_h, cudaStream_t stream,
                               const gpu_vision_weights_t *w,
                               const float *d_x, const float *d_residual, int n,
                               float *d_out, float *d_scratch);

// Upload one layer weights from CPU to GPU
bool gpu_vision_upload_layer(const vision_layer_weights_t *cpu,
                              gpu_vision_weights_t *gpu);

// Free GPU weights
void gpu_vision_free_layer(gpu_vision_weights_t *gpu);

#ifdef __cplusplus
}
#endif

#endif
