#ifndef WUBU_CUDA_KERNELS_H
#define WUBU_CUDA_KERNELS_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
#include <cuda_runtime.h>
#include <cublas_v2.h>
extern "C" {
#else
// Opaque handle types for C callers (CUDA types are C++ only)
typedef struct cublasContext *cublasHandle_t;
typedef struct CUstream_st *cudaStream_t;
#endif

// ================================================================
// cuBLAS-backed matmul: C[M,N] = A[M,K] @ B[K,N]
// B is stored row-major [K,N]. Result C is row-major [M,N].
// Internally uses cublasSgemm (column-major convention).
// Returns CUDA error code.
// ================================================================
int wubu_cuda_matmul(cublasHandle_t handle,
                     const float *A, int M, int K,
                     const float *B, int N,
                     float *C, float alpha, float beta);

// ================================================================
// Element-wise CUDA kernels (launched on current stream)
// ================================================================
void wubu_cuda_silu(int n, const float *x, float *y, cudaStream_t stream);
void wubu_cuda_sigmoid(int n, const float *x, float *y, cudaStream_t stream);
void wubu_cuda_softplus(int n, const float *x, float *y, cudaStream_t stream);
void wubu_cuda_exp(int n, const float *x, float *y, cudaStream_t stream);

// ================================================================
// L2 Normalization
// ================================================================
void wubu_cuda_l2_norm(int B, int T, int n_heads, int d,
                       const float *x, float eps, float *out,
                       cudaStream_t stream);

// ================================================================
// RMSNorm
// ================================================================
void wubu_cuda_rms_norm(int B, int T, int d,
                        const float *x, const float *weight, float eps,
                        float *out, cudaStream_t stream);

// ================================================================
// RMSNorm per-head (for GQA Q/K normalization)
// x: [total_heads, d] — each head normalized independently
// ================================================================
void wubu_cuda_rms_norm_heads(int total_heads, int d,
                               const float *x, const float *weight, float eps,
                               float *out, cudaStream_t stream);

// ================================================================
// Add bias to each row: out[i] = x[i] + bias[i % bias_len]
// For row-major [N, D] where bias is broadcast across rows
// ================================================================
void wubu_cuda_add_bias(int N, int D, const float *x,
                        const float *bias, float *out,
                        cudaStream_t stream);

// ================================================================
// Multiply each row by scalar: out[i] = x[i] * scalar[i % scalar_len]
// For row-major [N, D] where scalar is broadcast across rows
// ================================================================
void wubu_cuda_mul_by_scalar(int N, int D, const float *x,
                             const float *scalar, float *out,
                             cudaStream_t stream);

// ================================================================
// Gated normalization: SSM output gate
// delta_out[N, V_HEADS, D_STATE] — RMSNorm per head,
// multiply by norm_weight, then element-wise multiply by silu(z)
// ================================================================
void wubu_cuda_gated_norm(int B, int T, int n_heads, int d,
                          float *delta_out,
                          const float *norm_weight,
                          const float *z_silu,  // pre-computed SiLU(z)
                          cudaStream_t stream);

// ================================================================
// Strided split: copy contiguous slices from conv_out to Q/K/V buffers
// conv_out[N, C] → q_out[N, KEY_DIM], k_out[N, KEY_DIM], v_out[N, VALUE_DIM]
// where C = 2*KEY_DIM + VALUE_DIM
// ================================================================
void wubu_cuda_split_qkv(int N, int kdim, int vdim,
                         const float *conv_out,
                         float *q_out, float *k_out, float *v_out,
                         cudaStream_t stream);

// ================================================================
// Causal 1D Convolution (depthwise)
// ================================================================
void wubu_cuda_conv1d(int B, int T, int C, int k,
                      const float *input, const float *kernel,
                      float *output, cudaStream_t stream);

// ================================================================
// Gated Delta Net recurrence step (one head, one token)
// h: [D_STATE, D_STATE] mutable on GPU
// ================================================================
void wubu_cuda_delta_net_step(float *h,
                              const float *k_vh, const float *v_vh,
                              const float *q_vh,
                              float gate, float beta,
                              float *out_vh,
                              cudaStream_t stream);

// ================================================================
// Parallel associative scan for all heads over all tokens
// ================================================================
void wubu_cuda_ssm_parallel_scan(int B, int T,
    const float *d_q_norm,    // [N, K_HEADS, D_STATE]
    const float *d_k_norm,    // [N, K_HEADS, D_STATE]
    const float *d_v_conv,    // [N, V_HEADS, D_STATE]
    const float *d_gate,      // [N, DT_RANK]
    const float *d_beta,      // [N, DT_RANK]
    float *d_h_states,        // [B, V_HEADS, D_STATE, D_STATE]
    float *d_delta_out,       // [N, V_HEADS, D_STATE]
    cudaStream_t stream);

// ================================================================
// Fully fused SSM forward on GPU (single kernel for entire SSM layer)
// ================================================================
size_t wubu_cuda_ssm_forward_query_scratch(int B, int T);
void wubu_cuda_ssm_forward(cublasHandle_t handle, cudaStream_t stream,
    int B, int T,
    const float *d_x,
    const float *d_attn_qkv,
    const float *d_attn_gate,
    const float *d_ssm_beta,
    const float *d_ssm_alpha,
    const float *d_ssm_dt_bias,
    const float *d_ssm_a,
    const float *d_ssm_conv1d,
    const float *d_ssm_norm,
    const float *d_ssm_out,
    float *d_h_states,
    float *d_conv_state,
    float *d_output,
    float *d_scratch);

// ================================================================
// GQA Attention kernel (fused: Q/K RMSNorm + causal dot-product + softmax + V weighted sum + gate)
// ================================================================
void wubu_cuda_gqa_forward(cublasHandle_t handle, cudaStream_t stream,
    int B, int T,
    const float *d_Q_full,
    const float *d_K,
    const float *d_V,
    const float *d_Q_norm_w,
    const float *d_K_norm_w,
    const float *d_output_w,
    float *d_output,
    float *d_scratch,
    const float *d_sincos);

// Precompute rotary position frequencies (sin/cos) for RoPE
void wubu_cuda_precompute_rotary(int T, float *d_sincos, cudaStream_t stream);

// Apply RoPE to Q [N, Q_HEADS, HEAD_DIM] and K [N, KV_HEADS, HEAD_DIM] in-place
// Applies rotary to first ROTARY_DIM dimensions of each head.
void wubu_cuda_apply_rotary_to_qk(float *d_Q, float *d_K,
    int B, int T, int n_q_heads, int n_kv_heads, int head_dim,
    const float *d_sincos, cudaStream_t stream);

// GQA attention only (assumes Q/K already RMSNorm'd)
void wubu_cuda_gqa_attention_only(cublasHandle_t handle, cudaStream_t stream,
    int B, int T,
    const float *d_Q, const float *d_K, const float *d_V,
    float *d_output, int n_q_heads, int n_kv_heads, int head_dim);

// GQA gate multiply: x *= sigmoid(gate_part_of_Q_full)
void wubu_cuda_gqa_gate(float *d_x, const float *d_Q_full,
    int N, int q_dim, cudaStream_t stream);

// ================================================================
// Chunked attention with persistent KV cache (256K-capable)
//
// Q_chunk [C, N_Q_HEADS * HEAD_DIM] — already RMSNorm'd + RoPE'd
// K_cache [T_cache, N_KV_HEADS * HEAD_DIM] — already RMSNorm'd + RoPE'd
// V_cache [T_cache, N_KV_HEADS * HEAD_DIM]
// gate_full [C, N_Q_HEADS * HEAD_DIM] — raw gate (pre-sigmoid) from fused Q projection
//
// Output: d_out [C, D_MODEL] — after gate multiply + output projection
//
// d_score_scratch: [C * N_Q_HEADS * T_cache] or NULL (internal alloc).
// Returns required scratch bytes if d_score_scratch==NULL.
// ================================================================
size_t wubu_cuda_chunked_attn_query_scratch(int C, int T_max);
void wubu_cuda_chunked_attn(cublasHandle_t handle, cudaStream_t stream,
    int C, int T_cache,
    const float *d_Q_chunk,   // [C, N_Q_HEADS * HEAD_DIM] RMSNorm'd + RoPE'd
    const float *d_K_cache,   // [T_cache, N_KV_HEADS * HEAD_DIM] RMSNorm'd + RoPE'd
    const float *d_V_cache,   // [T_cache, N_KV_HEADS * HEAD_DIM]
    const float *d_gate_full, // [C, N_Q_HEADS * HEAD_DIM] raw gate
    const float *d_output_w,  // [N_Q_HEADS * HEAD_DIM, D_MODEL]
    float *d_out,             // [C, D_MODEL]
    float *d_score_scratch);  // [C * N_Q_HEADS * T_cache] or NULL

// ================================================================
// Poincaré ball hyperbolic CUDA kernels
// ================================================================
void wubu_cuda_norm(const float *d_in, float *d_out, int n_vecs, int dim, cudaStream_t stream);
void wubu_cuda_exp_map(const float *d_v, const float *d_norms, float R,
                        float *d_out, int n_vecs, int dim, cudaStream_t stream);
void wubu_cuda_log_map(const float *d_v, const float *d_norms, float R,
                        float *d_out, int n_vecs, int dim, cudaStream_t stream);
void wubu_cuda_mobius_scalar_mul(const float *d_v, const float *d_norms,
                                  float r, float R, float *d_out,
                                  int n_vecs, int dim, cudaStream_t stream);
void wubu_cuda_mobius_add(const float *d_x, const float *d_y,
                           const float *d_nx2, const float *d_ny2,
                           float *d_out, int n_vecs, int dim, cudaStream_t stream);

// Poincaré recurrence — replaces Euclidean step 9 in SSM forward
void wubu_cuda_poincare_recurrence(cublasHandle_t handle, cudaStream_t stream,
    int B, int T, float R,
    float *d_h_states,
    const float *d_q_norm, const float *d_k_norm, const float *d_v_conv,
    const float *d_gate, const float *d_beta,
    float *d_delta_out,
    float *d_states_t);

// ================================================================
// SSM scalar parallel associative scan (Mamba-style)
// ================================================================
void wubu_cuda_ssm_scalar_scan(int B_, int T, int d,
    const float *d_A,
    const float *d_B,
    const float *d_v,
    float *d_h,
    float *d_delta_out,
    cudaStream_t stream);

// ================================================================
// MoE dispatch: group tokens by expert, do batched matmuls on GPU
// ================================================================
size_t wubu_cuda_moe_dispatch_query_scratch(int B, int T);
void wubu_cuda_moe_dispatch(cublasHandle_t handle, cudaStream_t stream,
    int B, int T,
    const float *d_x,
    const int *d_assignments,
    const float *d_weights,
    const float *d_gate_exps,
    const float *d_up_exps,
    const float *d_down_exps,
    const float *d_gate_shexp,
    const float *d_up_shexp,
    const float *d_down_shexp,
    float *d_output,
    float *d_scratch);

// ================================================================
// CUDA context management
// ================================================================
bool wubu_cuda_init(cublasHandle_t *handle, cudaStream_t *stream);
void wubu_cuda_destroy(cublasHandle_t handle, cudaStream_t stream);
float *wubu_cuda_alloc(size_t n_bytes);
void wubu_cuda_free(float *ptr);
void wubu_cuda_to_device(const float *host, float *dev, size_t n_bytes, cudaStream_t stream);
void wubu_cuda_to_host(const float *dev, float *host, size_t n_bytes, cudaStream_t stream);

// ================================================================
// Softmax in-place over last dim: x[N, D] → softmax each row
// ================================================================
void wubu_cuda_softmax(float *x, int N, int D, cudaStream_t stream);

// ================================================================
// GPU MoE forward — single token, active experts, cuBLAS SGEMM
// x: [D_MODEL] on GPU
// gw/uw/dw: host pointers to dequantized FP32 expert weights
// out: [D_MODEL] GPU output
// tmp: scratch [D_FF * 3 + D_MODEL]
// ================================================================
void wubu_cuda_moe_fwd_1tok(cublasHandle_t handle, cudaStream_t stream,
    const float *d_x,
    const float *const*gw, const float *const*uw, const float *const*dw,
    const int *eids, const float *ewgts, int n_active,
    float *d_out,
    float *d_gate_w, float *d_up_w, float *d_silu,
    float *d_dn_w, float *d_contrib);

#ifdef __cplusplus
}
#endif

#endif // WUBU_CUDA_KERNELS_H
