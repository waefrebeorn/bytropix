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
//
// This replaces the host-loop delta_net_step with a single kernel
// that processes all T tokens for all 32 V-heads in one launch.
// Each block handles one V-head for one batch item.
//
// Inputs (prepared on GPU before launch):
//   q_norm:  [N, K_HEADS, D_STATE]   L2-normalized Q (repeat_factor maps K→V)
//   k_norm:  [N, K_HEADS, D_STATE]   L2-normalized K
//   v_conv:  [N, V_HEADS, D_STATE]   raw V from conv output (not L2'd)
//   gate:    [N, DT_RANK=32]         gate[t][h] = softplus(alpha+dt_bias) * A_log
//   beta:    [N, DT_RANK=32]         sigmoid(beta_raw)
//   repeat_factor = V_HEADS / K_HEADS = 2
//
// In/Out:
//   h_states: [N_BATCH, V_HEADS, D_STATE, D_STATE] initial state per batch item
//             updated to final state on return
//
// Output:
//   delta_out: [N, V_HEADS, D_STATE]  scan output (h[t] @ q[t] per head)
//
// N = B * T. B block dim for batch, grid = (B * V_HEADS) blocks.
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
//
// Processes one SSM layer: QKV proj → gate proj → beta/alpha → softplus
// → conv1d → split QKV → L2 norm → parallel scan → gated norm → output
// with NO host-level loops over tokens.
//
// x:        [B, T, D_MODEL]   — input to SSM layer
// weights:  SSM layer weight pointers on DEVICE
// h_states: [B, V_HEADS, D_STATE, D_STATE] — mutable state
// conv_state: [B, CONV_KERNEL-1, CONV_DIM] — mutable conv state
// output:   [B, T, D_MODEL]
// scratch:  device scratch buffer (size determined by formula)
// scratch_size: required scratch bytes (0 to query, then allocate)
//
// Returns required scratch_size on query (scratch=NULL).
// ================================================================
size_t wubu_cuda_ssm_forward_query_scratch(int B, int T);
void wubu_cuda_ssm_forward(cublasHandle_t handle, cudaStream_t stream,
    int B, int T,
    const float *d_x,
    const float *d_attn_qkv,    // [D_MODEL, CONV_DIM]
    const float *d_attn_gate,   // [D_MODEL, VALUE_DIM]
    const float *d_ssm_beta,    // [D_MODEL, DT_RANK]
    const float *d_ssm_alpha,   // [D_MODEL, DT_RANK]
    const float *d_ssm_dt_bias, // [DT_RANK]
    const float *d_ssm_a,       // [DT_RANK]
    const float *d_ssm_conv1d,  // [CONV_KERNEL, CONV_DIM]
    const float *d_ssm_norm,    // [SSM_D_STATE]
    const float *d_ssm_out,     // [VALUE_DIM, D_MODEL]
    float *d_h_states,          // [B, V_HEADS, D_STATE, D_STATE]
    float *d_conv_state,        // [B, CONV_KERNEL-1, CONV_DIM]
    float *d_output,
    float *d_scratch);

// ================================================================
// GQA Attention kernel (fused: Q/K RMSNorm + causal dot-product + softmax + V weighted sum + gate)
// Q_full:  [N, q_dim*2]  — first q_dim is Q, second q_dim is gate (pre-sigmoid)
// K/V:     [N, kv_dim]   — each [N, KV_HEADS * HEAD_DIM]
// q_norm_w: [HEAD_DIM], k_norm_w: [HEAD_DIM]
// mul_mask: multiplier[b*T+t] = 0 or other mask (not needed, handled by causal)
// out:     [N, q_dim]    — attention output before output projection
// ================================================================
void wubu_cuda_gqa_forward(cublasHandle_t handle, cudaStream_t stream,
    int B, int T,
    const float *d_Q_full,      // [N, q_dim*2]
    const float *d_K,           // [N, kv_dim]
    const float *d_V,           // [N, kv_dim]
    const float *d_Q_norm_w,    // [HEAD_DIM]
    const float *d_K_norm_w,    // [HEAD_DIM]
    const float *d_output_w,    // [q_dim, D_MODEL]
    float *d_output,            // [N, D_MODEL] — final output
    float *d_scratch,           // [N, q_dim] — temp for attention out
    const float *d_sincos);     // [T, ROTARY_DIM] — RoPE sin/cos table, or NULL to skip

// Precompute rotary position frequencies (sin/cos) for RoPE
// d_sincos: [T, ROTARY_DIM] — output buffer (caller allocates)
void wubu_cuda_precompute_rotary(int T, float *d_sincos, cudaStream_t stream);

// Apply RoPE to Q [N, Q_HEADS, HEAD_DIM] and K [N, KV_HEADS, HEAD_DIM] in-place
// Applies rotary to first ROTARY_DIM dimensions of each head.
// d_sincos: [T, ROTARY_DIM] — precomputed sin/cos (pair-duplicated for interleaved)
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
// Implements: h[t] = A[t] * h[t-1] + B[t] * v[t]
// Uses Blelloch-style parallel prefix scan over the associative
// operator (a,b) composed with (c,d) -> (a*c, a*d + b)
//
// A, B: [B_, T]     — scalar per-token A and B
// v:    [B_, T, d]  — input vectors
// h:    [B_, d]     — initial state (in) / final state (out)
// delta_out: [B_, T, d] — full scan output for all timesteps
// ================================================================
void wubu_cuda_ssm_scalar_scan(int B_, int T, int d,
    const float *d_A,        // [B_, T]
    const float *d_B,        // [B_, T]
    const float *d_v,        // [B_, T, d]
    float *d_h,              // [B_, d] — in/out
    float *d_delta_out,      // [B_, T, d]
    cudaStream_t stream);

// ================================================================
// MoE dispatch: group tokens by expert, do batched matmuls on GPU
//
// Takes token→expert assignments, builds an expert → token mapping
// via histogram + prefix sum, permutes tokens into an expert-grouped
// buffer, then does one cublasSgemm per expert (gate+up matmuls)
// followed by SiLU activation and down-projection.
//
// d_x: [B, T, D_MODEL] input tokens
// d_assignments: [B*T, N_ACTIVE_EXPTS] — expert indices (top-k per token)
// d_weights: [B*T, N_ACTIVE_EXPTS] — routing weights
// d_gate_exps: [D_MODEL, D_FF, N_EXPERTS] — expert gate weights
// d_up_exps:   [D_MODEL, D_FF, N_EXPERTS] — expert up weights
// d_down_exps: [D_FF, D_MODEL, N_EXPERTS] — expert down weights
// d_output: [B, T, D_MODEL] — output (shared expert + routed expert)
// d_scratch: workspace (size determined by query function)
//
// Returns required scratch size on query (d_scratch = NULL).
// ================================================================
size_t wubu_cuda_moe_dispatch_query_scratch(int B, int T);
void wubu_cuda_moe_dispatch(cublasHandle_t handle, cudaStream_t stream,
    int B, int T,
    const float *d_x,            // [B*T, D_MODEL]
    const int *d_assignments,    // [B*T, N_ACTIVE_EXPTS]
    const float *d_weights,      // [B*T, N_ACTIVE_EXPTS]
    const float *d_gate_exps,    // [D_MODEL, D_FF, N_EXPERTS]
    const float *d_up_exps,      // [D_MODEL, D_FF, N_EXPERTS]
    const float *d_down_exps,    // [D_FF, D_MODEL, N_EXPERTS]
    const float *d_gate_shexp,   // [D_MODEL, SHARED_D_FF]
    const float *d_up_shexp,     // [D_MODEL, SHARED_D_FF]
    const float *d_down_shexp,   // [SHARED_D_FF, D_MODEL]
    float *d_output,             // [B*T, D_MODEL]
    float *d_scratch);           // workspace or NULL to query

// ================================================================
// CUDA context management
// ================================================================
bool wubu_cuda_init(cublasHandle_t *handle, cudaStream_t *stream);
void wubu_cuda_destroy(cublasHandle_t handle, cudaStream_t stream);
float *wubu_cuda_alloc(size_t n_bytes);
void wubu_cuda_free(float *ptr);
void wubu_cuda_to_device(const float *host, float *dev, size_t n_bytes, cudaStream_t stream);
void wubu_cuda_to_host(const float *dev, float *host, size_t n_bytes, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // WUBU_CUDA_KERNELS_H
