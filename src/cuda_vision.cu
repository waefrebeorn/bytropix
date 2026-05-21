/**
 * cuda_vision.cu — GPU-accelerated ViT layer forward
 * cuBLAS Sgemm for all 4 linear projections + custom attention kernel.
 */
#include "cuda_vision.h"
#include <stdio.h>
#include <math.h>

#define CHECK_CUDA(call) do { cudaError_t e = call; if (e) { fprintf(stderr,"CUDA error %s:%d: %d\n",__FILE__,__LINE__,(int)e); return false; } } while(0)
#define CHECK_CUBLAS(call) do { cublasStatus_t s = call; if (s) { fprintf(stderr,"cuBLAS error %s:%d: %d\n",__FILE__,__LINE__,s); return false; } } while(0)

// ================================================================
// Fused attention kernel
// Q,K,V: [n, d] (d = V_HIDDEN = n_heads * head_dim)
// out: [n, d]
// ================================================================
__global__ void attention_kernel(const float *q, const float *k, const float *v,
                                  float *out, int n, int d, float scale) {
    int head = blockIdx.x;
    int row = threadIdx.x;
    if (head >= V_N_HEADS || row >= n) return;
    
    int head_dim = V_HEAD_DIM;
    const float *q_row = q + row * d + head * head_dim;
    float *out_row = out + row * d + head * head_dim;
    
    // Shared scores: use extern shared memory for arbitrary n
    extern __shared__ float scores[];
    float max_val = -1e30f;
    
    for (int t = 0; t < n; t++) {
        const float *k_t = k + t * d + head * head_dim;
        float sum = 0.0f;
        #pragma unroll
        for (int dd = 0; dd < head_dim; dd++)
            sum += q_row[dd] * k_t[dd];
        scores[t] = sum * scale;
        if (scores[t] > max_val) max_val = scores[t];
    }
    __syncthreads();
    
    // Softmax
    float sum_exp = 0.0f;
    for (int t = 0; t < n; t++) {
        scores[t] = expf(scores[t] - max_val);
        sum_exp += scores[t];
    }
    float inv_sum = 1.0f / (sum_exp + 1e-30f);
    for (int t = 0; t < n; t++)
        scores[t] *= inv_sum;
    __syncthreads();
    
    // Weighted sum of V
    for (int dd = 0; dd < head_dim; dd++) {
        float val = 0.0f;
        for (int t = 0; t < n; t++)
            val += scores[t] * v[t * d + head * head_dim + dd];
        out_row[dd] = val;
    }
}

// ================================================================
// GPU LayerNorm (per-row simplification)
// ================================================================
__global__ void layernorm_kernel(float *x, const float *w, const float *b,
                                  int n, int d, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    
    float *row_ptr = x + row * d;
    
    // Mean
    float mean = 0.0f;
    for (int i = 0; i < d; i++) mean += row_ptr[i];
    mean /= d;
    
    // Var
    float var = 0.0f;
    for (int i = 0; i < d; i++) var += (row_ptr[i] - mean) * (row_ptr[i] - mean);
    var /= d;
    
    float inv_std = rsqrtf(var + eps);
    for (int i = 0; i < d; i++)
        row_ptr[i] = (row_ptr[i] - mean) * inv_std * w[i] + b[i];
}

// ================================================================
// GELU activation (in-place)
// ================================================================
__global__ void gelu_kernel(float *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float c = 0.79788456f * (x[idx] + 0.044715f * x[idx] * x[idx] * x[idx]);
    x[idx] = 0.5f * x[idx] * (1.0f + tanhf(c));
}

// ================================================================
// Element-wise add: y = x + y (in-place on y)
// ================================================================
__global__ void add_kernel(const float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx] += x[idx];
}

// ================================================================
// Forward one ViT layer on GPU
// ================================================================
bool gpu_vision_layer_forward(cublasHandle_t cublas_h, cudaStream_t stream,
                               const gpu_vision_weights_t *w,
                               const float *d_x, int n,
                               float *d_out, float *d_scratch) {
    float alpha = 1.0f, beta = 0.0f;
    int d_model = V_HIDDEN;
    int d_qkv3 = V_HIDDEN * 3;
    float scale = 1.0f / sqrtf((float)V_HEAD_DIM);
    
    // Scratch layout: [d_qkv3, d_model*2]
    float *d_qkv = d_scratch;
    float *d_attn = d_scratch + n * d_qkv3;
    float *d_res  = d_attn + n * d_model;  // same as norm space
    
    // === Step 1: QKV projection ===
    // d_qkv[n, d_qkv3] = d_x[n, d_model] @ W[d_model, d_qkv3]^T
    CHECK_CUBLAS(cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                             d_qkv3, n, d_model, &alpha,
                             w->d_qkv_w, d_model,
                             d_x, d_model, &beta,
                             d_qkv, d_qkv3));
    // Add bias
    for (int i = 0; i < n; i++)
        CHECK_CUBLAS(cublasSaxpy(cublas_h, d_qkv3, &alpha, w->d_qkv_b, 1, d_qkv + i * d_qkv3, 1));
    
    // === Step 2: Multi-head attention ===
    float *d_q = d_qkv;
    float *d_k = d_qkv + n * d_model;
    float *d_v = d_qkv + n * d_model * 2;
    
    attention_kernel<<<V_N_HEADS, n > 256 ? 256 : n, n * sizeof(float), stream>>>(
        d_q, d_k, d_v, d_attn, n, d_model, scale);
    
    // === Step 3: Attention output projection ===
    // d_res[n, d_model] = d_attn[n, d_model] @ W_out[d_model, d_model]^T
    CHECK_CUBLAS(cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                             d_model, n, d_model, &alpha,
                             w->d_out_w, d_model,
                             d_attn, d_model, &beta,
                             d_res, d_model));
    for (int i = 0; i < n; i++)
        CHECK_CUBLAS(cublasSaxpy(cublas_h, d_model, &alpha, w->d_out_b, 1, d_res + i * d_model, 1));
    
    // === Step 4: Residual + LayerNorm (in-place on d_res, which is now d_norm) ===
    add_kernel<<<(n * d_model + 255) / 256, 256, 0, stream>>>(d_x, d_res, n * d_model);
    
    // === Step 5: FFN up projection ===
    // d_qkv[n, d_ffn] = d_res[n, d_model] @ W_up[d_model, d_ffn]^T (note: reusing d_qkv space as ffn buffer)
    // Actually use d_scratch beyond qkv space
    float *d_ffn_up = d_qkv;  // reuse qkv space since we no longer need it
    CHECK_CUBLAS(cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                             V_INTERMEDIATE, n, d_model, &alpha,
                             w->d_ffn_up_w, d_model,
                             d_res, d_model, &beta,
                             d_ffn_up, V_INTERMEDIATE));
    for (int i = 0; i < n; i++)
        CHECK_CUBLAS(cublasSaxpy(cublas_h, V_INTERMEDIATE, &alpha, w->d_ffn_up_b, 1, d_ffn_up + i * V_INTERMEDIATE, 1));
    
    // GELU
    gelu_kernel<<<(n * V_INTERMEDIATE + 255) / 256, 256, 0, stream>>>(d_ffn_up, n * V_INTERMEDIATE);
    
    // === Step 6: FFN down projection ===
    CHECK_CUBLAS(cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                             d_model, n, V_INTERMEDIATE, &alpha,
                             w->d_ffn_dn_w, V_INTERMEDIATE,
                             d_ffn_up, V_INTERMEDIATE, &beta,
                             d_out, d_model));
    for (int i = 0; i < n; i++)
        CHECK_CUBLAS(cublasSaxpy(cublas_h, d_model, &alpha, w->d_ffn_dn_b, 1, d_out + i * d_model, 1));
    
    // === Step 7: Second residual ===
    add_kernel<<<(n * d_model + 255) / 256, 256, 0, stream>>>(d_res, d_out, n * d_model);
    
    return true;
}

// ================================================================
// Upload layer weights to GPU
// ================================================================
bool gpu_vision_upload_layer(const vision_layer_weights_t *cpu,
                              gpu_vision_weights_t *gpu) {
    int d_model = V_HIDDEN;
    
    // QKV
    CHECK_CUDA(cudaMalloc(&gpu->d_qkv_w, d_model * d_model * 3 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(gpu->d_qkv_w, cpu->attn_qkv_weight,
                          d_model * d_model * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&gpu->d_qkv_b, d_model * 3 * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(gpu->d_qkv_b, cpu->attn_qkv_bias,
                          d_model * 3 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Out
    CHECK_CUDA(cudaMalloc(&gpu->d_out_w, d_model * d_model * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(gpu->d_out_w, cpu->attn_out_weight,
                          d_model * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&gpu->d_out_b, d_model * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(gpu->d_out_b, cpu->attn_out_bias,
                          d_model * sizeof(float), cudaMemcpyHostToDevice));
    
    // FFN up
    CHECK_CUDA(cudaMalloc(&gpu->d_ffn_up_w, d_model * V_INTERMEDIATE * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(gpu->d_ffn_up_w, cpu->ffn_up_weight,
                          d_model * V_INTERMEDIATE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&gpu->d_ffn_up_b, V_INTERMEDIATE * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(gpu->d_ffn_up_b, cpu->ffn_up_bias,
                          V_INTERMEDIATE * sizeof(float), cudaMemcpyHostToDevice));
    
    // FFN down
    CHECK_CUDA(cudaMalloc(&gpu->d_ffn_dn_w, V_INTERMEDIATE * d_model * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(gpu->d_ffn_dn_w, cpu->ffn_down_weight,
                          V_INTERMEDIATE * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&gpu->d_ffn_dn_b, d_model * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(gpu->d_ffn_dn_b, cpu->ffn_down_bias,
                          d_model * sizeof(float), cudaMemcpyHostToDevice));
    
    return true;
}

void gpu_vision_free_layer(gpu_vision_weights_t *gpu) {
    if (gpu->d_qkv_w) cudaFree(gpu->d_qkv_w);
    if (gpu->d_qkv_b) cudaFree(gpu->d_qkv_b);
    if (gpu->d_out_w) cudaFree(gpu->d_out_w);
    if (gpu->d_out_b) cudaFree(gpu->d_out_b);
    if (gpu->d_ffn_up_w) cudaFree(gpu->d_ffn_up_w);
    if (gpu->d_ffn_up_b) cudaFree(gpu->d_ffn_up_b);
    if (gpu->d_ffn_dn_w) cudaFree(gpu->d_ffn_dn_w);
    if (gpu->d_ffn_dn_b) cudaFree(gpu->d_ffn_dn_b);
    memset(gpu, 0, sizeof(*gpu));
}
