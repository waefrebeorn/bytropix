#include "cuda_kernels.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include <stdio.h>

// TGT-safe expf: clamp input to [-80, 80] to prevent overflow/underflow
static __device__ __inline__ float tgt_safe_expf(float x) {
    if (x > 80.0f) return expf(80.0f);
    if (x < -80.0f) return 0.0f;
    return expf(x);
}

// ================================================================
// cuBLAS matmul wrapper (handles row-major → column-major conversion)
// ================================================================
int wubu_cuda_matmul(cublasHandle_t handle,
                     const float *A, int M, int K,
                     const float *B, int N,
                     float *C, float alpha, float beta) {
    // A is [M,K] row-major, B is [K,N] row-major, C is [M,N] row-major
    // cuBLAS expects column-major: 
    // C_col[T] = A_col[T] @ B_col[T]  where T denotes transpose
    // C[M,N]@row = A[M,K]@row * B[K,N]@row
    // C_col = (B^T)[N,K] @ (A^T)[K,M]  => C_col is N×M col-major = C[M,N] row-major
    // cublasSgemm(handle, opB, opA, N, M, K, &alpha, B, N, A, K, &beta, C, N)
    // where C_col[N,M] = B[N,K] @ A[K,M] gives same elements as C[M,N] row-major
    return cublasSgemm(handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       N, M, K,
                       &alpha,
                       B, N,   // B is [K,N] → ld=N (leading dim of col-major is 2nd dim)
                       A, K,   // A is [M,K] → ld=K
                       &beta,
                       C, N);  // C is [M,N] → ld=N
}

// ================================================================
// Element-wise CUDA kernels
// ================================================================

__global__ void silu_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (v < -80.0f) y[i] = 0.0f;
    else y[i] = v / (1.0f + expf(-v));
}

__global__ void sigmoid_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (v < -80.0f) y[i] = 0.0f;
    else if (v > 80.0f) y[i] = 1.0f;
    else y[i] = 1.0f / (1.0f + expf(-v));
}

__global__ void softplus_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (v > 80.0f) y[i] = v;
    else if (v < -80.0f) y[i] = 0.0f;
    else y[i] = logf(1.0f + expf(v));
}

__global__ void exp_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = tgt_safe_expf(x[i]);
}

void wubu_cuda_silu(int n, const float *x, float *y, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    silu_kernel<<<grid, block, 0, stream>>>(x, y, n);
}

void wubu_cuda_sigmoid(int n, const float *x, float *y, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    sigmoid_kernel<<<grid, block, 0, stream>>>(x, y, n);
}

// Init constant value kernels
__global__ void init_neg_inf(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = -1e30f;
}
__global__ void init_zero_kernel(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = 0.0f;
}

void wubu_cuda_softplus(int n, const float *x, float *y, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    softplus_kernel<<<grid, block, 0, stream>>>(x, y, n);
}

void wubu_cuda_exp(int n, const float *x, float *y, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    exp_kernel<<<grid, block, 0, stream>>>(x, y, n);
}

// ================================================================
// L2 Normalization kernel
// Input: x [B, T, n_heads, d] stored [N, n_heads, d] where N = B*T
// Output: out same shape, each head normalized to unit L2 norm
// ================================================================
__global__ void l2_norm_kernel(const float *x, float *out, int N, int n_heads, int d, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * n_heads;
    if (idx >= total) return;

    const float *inp = x + idx * d;
    float *oup = out + idx * d;

    float sum_sq = 0.0f;
    for (int i = 0; i < d; i++) sum_sq += inp[i] * inp[i];

    float scale = rsqrtf(sum_sq + eps);
    for (int i = 0; i < d; i++) oup[i] = inp[i] * scale;
}

void wubu_cuda_l2_norm(int B, int T, int n_heads, int d,
                       const float *x, float eps, float *out,
                       cudaStream_t stream) {
    int N = B * T;
    int total = N * n_heads;
    int block = 256;
    int grid = (total + block - 1) / block;
    l2_norm_kernel<<<grid, block, 0, stream>>>(x, out, N, n_heads, d, eps);
}

// ================================================================
// RMSNorm per-head kernel (for GQA Q/K normalization)
// Input: x [B, n_heads, d], weight [d]
// Output: out [B, n_heads, d]
// Each head independently: out[i*hd + k] = x[i*hd + k] * weight[k] / sqrt(mean(x²) + eps)
// ================================================================
__global__ void rms_norm_heads_kernel(const float *x, const float *weight, float *out,
                                       int total_heads, int d, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_heads) return;

    const float *inp = x + idx * d;
    float *oup = out + idx * d;

    float sum_sq = 0.0f;
    for (int i = 0; i < d; i++) sum_sq += inp[i] * inp[i];
    float scale = rsqrtf(sum_sq / d + eps);
    for (int i = 0; i < d; i++) oup[i] = inp[i] * scale * weight[i];
}

void wubu_cuda_rms_norm_heads(int total_heads, int d,
                               const float *x, const float *weight, float eps,
                               float *out, cudaStream_t stream) {
    int block = 256;
    int grid = (total_heads + block - 1) / block;
    rms_norm_heads_kernel<<<grid, block, 0, stream>>>(x, weight, out, total_heads, d, eps);
}
__global__ void rms_norm_kernel(const float *x, const float *weight, float *out,
                                int N, int d, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float *inp = x + idx * d;
    float *oup = out + idx * d;
    
    // Double precision for sum-of-squares to avoid overflow with large values
    double sum_sq = 0.0;
    for (int i = 0; i < d; i++) {
        sum_sq += (double)inp[i] * (double)inp[i];
    }
    float rms = sqrtf((float)(sum_sq / d) + eps);
    float scale = 1.0f / rms;

    for (int i = 0; i < d; i++) oup[i] = inp[i] * scale * weight[i];
}
void wubu_cuda_rms_norm(int B, int T, int d,
                        const float *x, const float *weight, float eps,
                        float *out, cudaStream_t stream) {
    int N = B * T;
    int block = 256;
    int grid = (N + block - 1) / block;
    rms_norm_kernel<<<grid, block, 0, stream>>>(x, weight, out, N, d, eps);
}

// ================================================================
// Add bias broadcast kernel
// out[i] = x[i] + bias[i % D]
// ================================================================
__global__ void add_bias_kernel(const float *x, const float *bias,
                                float *out, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * D) return;
    out[i] = x[i] + bias[i % D];
}

void wubu_cuda_add_bias(int N, int D, const float *x,
                        const float *bias, float *out,
                        cudaStream_t stream) {
    int total = N * D;
    int block = 256;
    int grid = (total + block - 1) / block;
    add_bias_kernel<<<grid, block, 0, stream>>>(x, bias, out, N, D);
}

// ================================================================
// Multiply by scalar broadcast kernel
// out[i] = x[i] * scalar[i % D]
// ================================================================
__global__ void mul_by_scalar_kernel(const float *x, const float *scalar,
                                     float *out, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * D) return;
    out[i] = x[i] * scalar[i % D];
}

void wubu_cuda_mul_by_scalar(int N, int D, const float *x,
                             const float *scalar, float *out,
                             cudaStream_t stream) {
    int total = N * D;
    int block = 256;
    int grid = (total + block - 1) / block;
    mul_by_scalar_kernel<<<grid, block, 0, stream>>>(x, scalar, out, N, D);
}

// ================================================================
// Gated normalization: RMSNorm per head + multiply by silu(z)
// delta_out: [B, T, n_heads, d] — RMSNorm'd in-place
// norm_weight: [d] — per-dimension scale
// z_silu: [B, T, VALUE_DIM] — pre-computed SiLU(z) where VALUE_DIM = n_heads * d
//
// For each (b, t, h): 
//   vec = delta_out[bt, h, :]
//   rms = sqrt(mean(vec^2) + eps)
//   vec = (vec / rms) * norm_weight * z_silu[bt, h, :]
// ================================================================
__global__ void gated_norm_kernel(float *delta_out,
                                  const float *norm_weight,
                                  const float *z_silu,
                                  int B, int T, int n_heads, int d,
                                  float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T;
    int total = N * n_heads;
    if (idx >= total) return;

    float *vec = delta_out + idx * d;
    const float *z = z_silu + idx * d;

    // RMSNorm
    float sum_sq = 0.0f;
    for (int i = 0; i < d; i++) sum_sq += vec[i] * vec[i];
    float rms = sqrtf(sum_sq / d + eps);
    float scale = 1.0f / rms;

    // Apply norm weight and multiply by z
    for (int i = 0; i < d; i++) {
        vec[i] = vec[i] * scale * norm_weight[i] * z[i];
    }
}

void wubu_cuda_gated_norm(int B, int T, int n_heads, int d,
                          float *delta_out,
                          const float *norm_weight,
                          const float *z_silu,
                          cudaStream_t stream) {
    int N = B * T;
    int total = N * n_heads;
    int block = 256;
    int grid = (total + block - 1) / block;
    gated_norm_kernel<<<grid, block, 0, stream>>>(delta_out, norm_weight, z_silu,
                                                   B, T, n_heads, d, 1e-6f);
}

// ================================================================
// Split Q/K/V from conv output
// conv_out: [N, 2*KEY_DIM + VALUE_DIM]
// q_out:    [N, KEY_DIM]   — first KEY_DIM elements
// k_out:    [N, KEY_DIM]   — middle KEY_DIM elements
// v_out:    [N, VALUE_DIM] — last VALUE_DIM elements
// ================================================================
__global__ void split_qkv_kernel(const float *conv_out,
                                  float *q_out, float *k_out, float *v_out,
                                  int N, int kdim, int vdim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int C = 2 * kdim + vdim;
    if (idx >= N * C) return;

    int s = idx / C;
    int c = idx % C;
    int row_offset = s * C;

    if (c < kdim) {
        q_out[s * kdim + c] = conv_out[row_offset + c];
    } else if (c < 2 * kdim) {
        k_out[s * kdim + (c - kdim)] = conv_out[row_offset + c];
    } else {
        v_out[s * vdim + (c - 2 * kdim)] = conv_out[row_offset + c];
    }
}

void wubu_cuda_split_qkv(int N, int kdim, int vdim,
                         const float *conv_out,
                         float *q_out, float *k_out, float *v_out,
                         cudaStream_t stream) {
    int C = 2 * kdim + vdim;
    int total = N * C;
    int block = 256;
    int grid = (total + block - 1) / block;
    split_qkv_kernel<<<grid, block, 0, stream>>>(conv_out, q_out, k_out, v_out,
                                                   N, kdim, vdim);
}

// ================================================================
// 1D Convolution (depthwise, causal) kernel
// input: [B, T+k-1, C] padded (first k-1 elements are zeros or state)
// kernel: [k, C] — depthwise: each channel has its own kernel
// output: [B, T, C]
// ================================================================
__global__ void conv1d_kernel(const float *input, const float *kernel,
                              float *output, int B, int T, int C, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * C;
    if (idx >= total) return;

    int c = idx % C;
    int t = (idx / C) % T;
    int b = idx / (T * C);

    int inp_offset = (b * (T + k - 1) + t) * C + c;
    float sum = 0.0f;
    for (int ki = 0; ki < k; ki++) {
        sum += input[inp_offset + ki * C] * kernel[ki * C + c];
    }
    output[idx] = sum;
}

void wubu_cuda_conv1d(int B, int T, int C, int k,
                      const float *input, const float *kernel,
                      float *output, cudaStream_t stream) {
    int total = B * T * C;
    int block = 256;
    int grid = (total + block - 1) / block;
    conv1d_kernel<<<grid, block, 0, stream>>>(input, kernel, output, B, T, C, k);
}

// ================================================================
// Gated Delta Net step — matches CPU reference exactly
//
// CPU reference (wubu_ssm.c lines 316-352):
//   1. h *= gg (decay first: h[i][j] *= exp(gate))
//   2. hk[i] = sum_j h[i,j] * k_vh[j]  (uses DECAYED h)
//   3. diff[i] = v_vh[i] - hk[i]
//   4. h[i,j] += k_vh[i] * diff[j] * beta
//   5. out[i] = sum_j h[i,j] * q_vh[j]
// ================================================================
__global__ void delta_net_step_kernel(float *h,
                                      const float *k_vh,
                                      const float *v_vh,
                                      const float *q_vh,
                                      float gate_raw, float beta,
                                      float *out_vh,
                                      int d) {
    extern __shared__ float shared[];
    float *hk = shared;       // [D_STATE] — h @ k for each dim
    float *diff = shared + d;  // [D_STATE] — v - hk

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;

    float egate = tgt_safe_expf(gate_raw);

    // Phase 1: decay h, then compute hk[i] = sum_j h[i,j] * k[j]
    // Order matters: decay BEFORE hk (matches CPU)
    float hk_i = 0.0f;
    for (int j = 0; j < d; j++) {
        float hij = h[i * d + j] * egate;  // decay
        h[i * d + j] = hij;                // store decayed
        hk_i += hij * k_vh[j];             // hk from decayed h
    }
    hk[i] = hk_i;
    __syncthreads();

    // Phase 2: diff[i] = v_vh[i] - hk[i]
    diff[i] = v_vh[i] - hk[i];
    __syncthreads();

    // Phase 3: h[i,j] += k_vh[i] * diff[j] * beta
    // k_vh[i] is thread-local, diff[j] is in shared memory
    for (int j = 0; j < d; j++) {
        h[i * d + j] += k_vh[i] * diff[j] * beta;
    }
    __syncthreads();

    // Phase 4: out_vh[i] = sum_j h[i,j] * q_vh[j]
    float oi = 0.0f;
    for (int j = 0; j < d; j++) {
        oi += h[i * d + j] * q_vh[j];
    }
    out_vh[i] = oi;
}

void wubu_cuda_delta_net_step(float *h,
                              const float *k_vh, const float *v_vh,
                              const float *q_vh,
                              float gate, float beta,
                              float *out_vh,
                              cudaStream_t stream) {
    const int D = 128;  // SSM_D_STATE
    int block = D;      // one thread per state dimension
    int grid = 1;       // one head
    size_t shared = 2 * D * sizeof(float);  // hk + diff
    delta_net_step_kernel<<<grid, block, shared, stream>>>(
        h, k_vh, v_vh, q_vh, gate, beta, out_vh, D);
}

// ================================================================
// PARALLEL ASSOCIATIVE SCAN — Gated Delta Net Recurrence
//
// Processes all T tokens for one V-head in a single kernel.
// Uses the linear recurrence structure:
//   h[t] = A[t] * h[t-1] + B[t]
// where A[t] = exp(gate[t]) (scalar decay)
//       B[t] = k[t] ⊗ (v[t] * beta[t] - k[t]^T @ h[t-1] * beta[t])
//
// Phase 1: Token-loop sequential within each head (but all 32 heads in parallel)
// Phase 2: State materialization for all T tokens (for backprop)
// ================================================================

// One block per (batch, V-head). Each block has D_STATE=128 threads.
// Each thread handles one row i of the d×d state matrix h.
// State is kept in per-thread registers across the T-loop.
// Only diff[d] needs shared memory for the outer update.
__global__ void ssm_parallel_scan_kernel(
    const float *q_norm,      // [B, T, K_HEADS, D_STATE] — K-headed Q
    const float *k_norm,      // [B, T, K_HEADS, D_STATE] — K-headed K
    const float *v_conv,      // [B, T, V_HEADS, D_STATE] — V-headed V (raw)
    const float *gate,        // [B, T, DT_RANK] — gate per V-head
    const float *beta,        // [B, T, DT_RANK] — beta per V-head
    float *h_states,          // [B, V_HEADS, D_STATE, D_STATE] — initialized state
    float *delta_out,         // [B, T, V_HEADS, D_STATE] — output = h[t] @ q[t]
    int B, int T, int n_kheads, int n_vheads,
    int d, int repeat_factor) {

    // Each block: batch=blockIdx.x / n_vheads, vhead=blockIdx.x % n_vheads
    int b = blockIdx.x / n_vheads;
    int vh = blockIdx.x % n_vheads;
    int kh = vh / repeat_factor;  // which K-head maps to this V-head

    // Shared memory: diff[d] only — d floats = 512 bytes (fits in 48KB default)
    extern __shared__ float sh_diff[];

    // Thread i handles row i of the d×d state matrix
    int i = threadIdx.x;
    if (i >= d) return;

    // Load initial state from global — each thread loads its row
    float *state_base = h_states + ((b * n_vheads + vh) * d * d);
    
    // Local array for the thread's row of the state matrix
    // Compiler may spill to local memory (L1-cached) which is fine
    float h_row[128];
    
    for (int j = 0; j < d; j++) {
        h_row[j] = state_base[i * d + j];
    }

    // Token loop — sequential within block (T iterations, each does d² work)
    // Total work: T * d² = T * 16384 flops per head = ~67Mflops for T=4096
    for (int t = 0; t < T; t++) {
        int s = b * T + t;  // flat index into [N, ...] arrays

        // Load q, k, v for this token and head
        const float *q_kh = q_norm + (s * n_kheads + kh) * d;
        const float *k_kh = k_norm + (s * n_kheads + kh) * d;
        const float *v_vh = v_conv + (s * n_vheads + vh) * d;

        float gate_val = gate[s * DT_RANK + vh];
        float beta_val = beta[s * DT_RANK + kh];

        // Load q[i], k[i], v[i] for this thread
        float qi = q_kh[i];
        float ki = k_kh[i];
        float vi = v_vh[i];

        // Step 1: Decay state
        float egate = tgt_safe_expf(gate_val);
        for (int j = 0; j < d; j++) {
            h_row[j] *= egate;
        }

        // Step 2: Compute hk[i] = sum_j h[i][j] * k[j]
        // This is a reduction: thread i computes dot(row_i, k_full)
        float hk_i = 0.0f;
        for (int j = 0; j < d; j++) {
            hk_i += h_row[j] * k_kh[j];
        }

        // Step 3: diff[i] = v[i] - hk[i] — share via shmem
        sh_diff[i] = vi - hk_i;
        __syncthreads();

        // Step 4: Outer update: h[i][j] += k[i] * diff[j] * beta
        for (int j = 0; j < d; j++) {
            h_row[j] += ki * sh_diff[j] * beta_val;
        }
        __syncthreads();

        // Step 5: Output = h[i] @ q
        float out_i = 0.0f;
        for (int j = 0; j < d; j++) {
            out_i += h_row[j] * q_kh[j];
        }

        // Write delta_out
        delta_out[(s * n_vheads + vh) * d + i] = out_i;
    }

    // Write final state back
    for (int j = 0; j < d; j++) {
        state_base[i * d + j] = h_row[j];
    }
}

void wubu_cuda_ssm_parallel_scan(int B, int T,
    const float *d_q_norm,
    const float *d_k_norm,
    const float *d_v_conv,
    const float *d_gate,
    const float *d_beta,
    float *d_h_states,
    float *d_delta_out,
    cudaStream_t stream) {

    const int d = SSM_D_STATE;  // 128
    const int n_blocks = B * SSM_V_HEADS;  // one block per (batch, V-head)

    // Shared memory: diff[d] only = 128 floats = 512 bytes << 48KB default
    size_t shared_bytes = d * sizeof(float);

    ssm_parallel_scan_kernel<<<n_blocks, d, shared_bytes, stream>>>(
        d_q_norm, d_k_norm, d_v_conv, d_gate, d_beta,
        d_h_states, d_delta_out,
        B, T, SSM_K_HEADS, SSM_V_HEADS, d, SSM_V_HEADS / SSM_K_HEADS);
}

// ================================================================
// Fused SSM layer forward — all steps on GPU, no host loops
// ================================================================

size_t wubu_cuda_ssm_forward_query_scratch(int B, int T) {
    const int N = B * T;
    // Largest temps:
    // qkv_all: [N, CONV_DIM] = N*8192 floats
    // z_all:   [N, VALUE_DIM] = N*4096 floats
    // beta_raw/gate: [N, DT_RANK] = N*32 floats each
    // conv_input: [B, T+3, CONV_DIM] = B*(T+3)*8192
    // conv_output: [N, CONV_DIM] = N*8192
    // q/k/v conv/norm: ~ 5 * N*VALUE_DIM = 5*N*4096
    // delta_out: [N, VALUE_DIM]
    // z_silu: [N, VALUE_DIM]
    // alpha_raw, alpha_biased, alpha_softplus: 3*N*32
    //
    // Max is roughly: 3*N*8192 + 5*N*4096 + 3*N*32 ≈ N*(24576+20480+96) ≈ 45K*N
    // Round up generously
    size_t per_token = (size_t)(CONV_DIM * 3 + VALUE_DIM * 6 + DT_RANK * 10);
    return (size_t)N * per_token * sizeof(float) + 1024*1024;  // +1MB padding
}

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
    float *d_scratch) {

    const int N = B * T;
    const int C = CONV_DIM;      // 8192
    const int kdim = KEY_DIM;    // 2048
    const int vdim = VALUE_DIM;  // 4096
    const int dr = DT_RANK;      // 32

    // Scratch layout (offsets in floats):
    size_t offset = 0;
    float *d_qkv_all    = d_scratch + offset; offset += N * C;
    float *d_z_all      = d_scratch + offset; offset += N * vdim;
    float *d_beta_raw   = d_scratch + offset; offset += N * dr;
    float *d_alpha_raw  = d_scratch + offset; offset += N * dr;
    float *d_beta_final = d_scratch + offset; offset += N * dr;
    float *d_gate_final = d_scratch + offset; offset += N * dr;
    float *d_alpha_biased= d_scratch + offset; offset += N * dr;
    float *d_alpha_softplus= d_scratch + offset; offset += N * dr;
    float *d_conv_input  = d_scratch + offset; offset += B * (T + CONV_KERNEL - 1) * C;
    float *d_conv_output = d_scratch + offset; offset += N * C;
    float *d_q_conv = d_scratch + offset; offset += N * kdim;
    float *d_k_conv = d_scratch + offset; offset += N * kdim;
    float *d_v_conv = d_scratch + offset; offset += N * vdim;
    float *d_q_norm = d_scratch + offset; offset += N * kdim;
    float *d_k_norm = d_scratch + offset; offset += N * kdim;
    float *d_delta_out = d_scratch + offset; offset += N * vdim;
    float *d_z_silu     = d_scratch + offset; offset += N * vdim;
    // plus some extra for gated_norm intermediate
    float *d_gated_out  = d_scratch + offset;

    // Step 1: QKV projection — x @ attn_qkv^T -> qkv_all[N, C]
    // attn_qkv is [D_MODEL, C] stored row-major
    // C[N, C] = A[N, D] @ B[D, C] => matmul(handle, A, N, D, B, C, C, 1, 0)
    wubu_cuda_matmul(handle, d_x, N, D_MODEL, d_attn_qkv, C, d_qkv_all, 1.0f, 0.0f);

    // Step 2: Gate projection — x @ attn_gate^T -> z_all[N, vdim]
    wubu_cuda_matmul(handle, d_x, N, D_MODEL, d_attn_gate, vdim, d_z_all, 1.0f, 0.0f);

    // Step 3: Beta/Alpha projections
    // beta_raw[N, dr] = x[N, D] @ ssm_beta[D, dr]^T
    wubu_cuda_matmul(handle, d_x, N, D_MODEL, d_ssm_beta, dr, d_beta_raw, 1.0f, 0.0f);
    // alpha_raw[N, dr] = x[N, D] @ ssm_alpha[D, dr]^T
    wubu_cuda_matmul(handle, d_x, N, D_MODEL, d_ssm_alpha, dr, d_alpha_raw, 1.0f, 0.0f);

    // Step 4: Compute beta = sigmoid(beta_raw), alpha_biased = alpha + dt_bias
    wubu_cuda_sigmoid(N * dr, d_beta_raw, d_beta_final, stream);
    wubu_cuda_add_bias(N, dr, d_alpha_raw, d_ssm_dt_bias, d_alpha_biased, stream);
    // softplus on alpha_biased
    wubu_cuda_softplus(N * dr, d_alpha_biased, d_alpha_softplus, stream);
    // gate = alpha_softplus * A_log
    wubu_cuda_mul_by_scalar(N, dr, d_alpha_softplus, d_ssm_a, d_gate_final, stream);

    // Step 5: Build conv_input from conv_state + qkv_all
    // conv_input[b, t_start..t_end, c]
    // For each batch: first (k-1) elements from conv_state, rest from qkv_all
    {
        int k_1 = CONV_KERNEL - 1;  // 3
        int row_stride = (T + k_1) * C;
        for (int b = 0; b < B; b++) {
            // Copy conv_state: d_conv_state[b, :, :] -> conv_input[b, 0:k_1, :]
            wubu_cuda_to_device(
                (const float*)((size_t)d_conv_state + b * k_1 * C * sizeof(float)),
                (float*)((size_t)d_conv_input + b * row_stride * sizeof(float)),
                k_1 * C * sizeof(float), stream);
            // Copy qkv_all[b, :, :] -> conv_input[b, k_1:, :]
            // Use cudaMemcpy2DAsync for 2D copy: each row is C floats
            // Actually simpler: manual kernel, but let's use cudaMemcpyAsync with offsets
            size_t src_offset = b * T * C * sizeof(float);
            size_t dst_offset = b * row_stride * sizeof(float) + k_1 * C * sizeof(float);
            cudaMemcpyAsync((char*)d_conv_input + dst_offset,
                          (const char*)d_qkv_all + src_offset,
                          T * C * sizeof(float),
                          cudaMemcpyDeviceToDevice, stream);
        }
    }

    // Step 6: Conv1d
    wubu_cuda_conv1d(B, T, C, CONV_KERNEL, d_conv_input, d_ssm_conv1d, d_conv_output, stream);

    // Step 7: SiLU on conv output
    wubu_cuda_silu(N * C, d_conv_output, d_conv_output, stream);

    // Step 8: Update conv_state — last k-1 elements of input
    {
        int k_1 = CONV_KERNEL - 1;
        int row_stride = (T + k_1) * C;
        for (int b = 0; b < B; b++) {
            // conv_input[b, T:, :] -> conv_state[b, :, :]
            size_t src_offset = b * row_stride * sizeof(float) + T * C * sizeof(float);
            cudaMemcpyAsync((char*)d_conv_state + b * k_1 * C * sizeof(float),
                          (const char*)d_conv_input + src_offset,
                          k_1 * C * sizeof(float),
                          cudaMemcpyDeviceToDevice, stream);
        }
    }

    // Step 9: Split QKV: conv_output -> q_conv, k_conv, v_conv
    wubu_cuda_split_qkv(N, kdim, vdim, d_conv_output, d_q_conv, d_k_conv, d_v_conv, stream);

    // Step 10: L2 normalize Q and K
    wubu_cuda_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, d_q_conv, 1e-12f, d_q_norm, stream);
    wubu_cuda_l2_norm(B, T, SSM_K_HEADS, SSM_D_STATE, d_k_conv, 1e-12f, d_k_norm, stream);

    // Step 11: Parallel associative scan (replaces host loop)
    wubu_cuda_ssm_parallel_scan(B, T,
        d_q_norm, d_k_norm, d_v_conv,
        d_gate_final, d_beta_final,
        d_h_states, d_delta_out,
        stream);

    // Step 12: z = SiLU(z_all)
    wubu_cuda_silu(N * vdim, d_z_all, d_z_silu, stream);

    // Step 13: Gated norm: delta_out * rms_norm * z_silu
    wubu_cuda_gated_norm(B, T, SSM_V_HEADS, SSM_D_STATE,
                         d_delta_out, d_ssm_norm, d_z_silu, stream);

    // Step 14: Output projection: delta_out[N, vdim] @ ssm_out[vdim, D_MODEL]^T -> output[N, D_MODEL]
    // delta_out is now gated-normalized (same memory, modified in-place)
    // Actually gated_norm modifies delta_out in-place, so it's [N, vdim]
    wubu_cuda_matmul(handle, d_delta_out, N, vdim, d_ssm_out, D_MODEL, d_output, 1.0f, 0.0f);
}

// ================================================================
// CUDA context management
// ================================================================

bool wubu_cuda_init(cublasHandle_t *handle, cudaStream_t *stream) {
    cudaError_t ce;
    cublasStatus_t cs;

    ce = cudaSetDevice(0);
    if (ce != cudaSuccess) {
        fprintf(stderr, "CUDA: cudaSetDevice(0) failed: %s\n", cudaGetErrorString(ce));
        return false;
    }

    ce = cudaStreamCreate(stream);
    if (ce != cudaSuccess) {
        fprintf(stderr, "CUDA: cudaStreamCreate failed: %s\n", cudaGetErrorString(ce));
        return false;
    }

    cs = cublasCreate(handle);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUDA: cublasCreate failed\n");
        return false;
    }

    cs = cublasSetStream(*handle, *stream);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUDA: cublasSetStream failed\n");
        return false;
    }

    return true;
}

void wubu_cuda_destroy(cublasHandle_t handle, cudaStream_t stream) {
    if (handle) cublasDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
}

float *wubu_cuda_alloc(size_t n_bytes) {
    float *ptr = NULL;
    cudaError_t ce = cudaMalloc(&ptr, n_bytes);
    if (ce != cudaSuccess) {
        fprintf(stderr, "CUDA: cudaMalloc(%zu) failed: %s\n", n_bytes, cudaGetErrorString(ce));
        return NULL;
    }
    return ptr;
}

void wubu_cuda_free(float *ptr) {
    if (ptr) cudaFree(ptr);
}

void wubu_cuda_to_device(const float *host, float *dev, size_t n_bytes, cudaStream_t stream) {
    cudaMemcpyAsync(dev, host, n_bytes, cudaMemcpyHostToDevice, stream);
}

void wubu_cuda_to_host(const float *dev, float *host, size_t n_bytes, cudaStream_t stream) {
    cudaMemcpyAsync(host, dev, n_bytes, cudaMemcpyDeviceToHost, stream);
}

// ================================================================
// RoPE: Rotary Position Embedding (from Qwen3.6 config.json)
//   rope_theta = 10,000,000
//   partial_rotary_factor = 0.25 → ROTARY_DIM = 64 (out of 256 head_dim)
// ================================================================

// Precompute sin/cos for positions 0..T-1, dimension 0..ROTARY_DIM-1
// Output layout: sincos[pos * ROTARY_DIM + i] = sin(pos * theta_i) for even i, cos for odd i
// Where theta_i = rope_theta^(-2*floor(i/2)/ROTARY_DIM)
__global__ void precompute_rotary_kernel(float *sincos, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = T * ROTARY_DIM;
    if (idx >= total) return;
    int pos = idx / ROTARY_DIM;
    int dim = idx % ROTARY_DIM;
    int half = dim / 2;
    float theta = powf(ROPE_THETA, -2.0f * half / ROTARY_DIM);
    float angle = pos * theta;
    // For each pair (2i, 2i+1): dim%2==0 → sin, dim%2==1 → cos
    sincos[idx] = (dim % 2 == 0) ? sinf(angle) : cosf(angle);
}

void wubu_cuda_precompute_rotary(int T, float *d_sincos, cudaStream_t stream) {
    int total = T * ROTARY_DIM;
    int block = 256;
    int grid = (total + block - 1) / block;
    precompute_rotary_kernel<<<grid, block, 0, stream>>>(d_sincos, T);
}

// Apply RoPE to Q and K in-place.
// Q layout: [B, T, n_q_heads, head_dim]
// K layout: [B, T, n_kv_heads, head_dim]
// sincos layout: [T, ROTARY_DIM] — for each (pos, dim): even=sin, odd=cos
// Only first ROTARY_DIM dimensions of each head are rotated.
__global__ void apply_rotary_qk_kernel(float *Q, float *K,
    int B, int T, int n_q_heads, int n_kv_heads, int head_dim,
    const float *sincos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T;
    if (idx >= total) return;
    int b = idx / T;
    int t = idx % T;
    const float *sp = sincos + t * ROTARY_DIM;

    // Apply to Q: each (b,t) has n_q_heads heads of head_dim
    float *q_base = Q + ((b * T + t) * n_q_heads) * head_dim;
    for (int h = 0; h < n_q_heads; h++) {
        float *qv = q_base + h * head_dim;
        for (int d = 0; d < ROTARY_DIM; d += 2) {
            float x0 = qv[d];
            float x1 = qv[d+1];
            float sin_val = sp[d];
            float cos_val = sp[d+1];
            qv[d]   = x0 * cos_val - x1 * sin_val;
            qv[d+1] = x0 * sin_val + x1 * cos_val;
        }
        // remaining dims (ROTARY_DIM..head_dim-1) are unchanged
    }

    // Apply to K: each (b,t) has n_kv_heads heads of head_dim
    float *k_base = K + ((b * T + t) * n_kv_heads) * head_dim;
    for (int h = 0; h < n_kv_heads; h++) {
        float *kv = k_base + h * head_dim;
        for (int d = 0; d < ROTARY_DIM; d += 2) {
            float x0 = kv[d];
            float x1 = kv[d+1];
            float sin_val = sp[d];
            float cos_val = sp[d+1];
            kv[d]   = x0 * cos_val - x1 * sin_val;
            kv[d+1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

void wubu_cuda_apply_rotary_to_qk(float *d_Q, float *d_K,
    int B, int T, int n_q_heads, int n_kv_heads, int head_dim,
    const float *d_sincos, cudaStream_t stream) {
    int total = B * T;
    int block = 128;
    int grid = (total + block - 1) / block;
    apply_rotary_qk_kernel<<<grid, block, 0, stream>>>(
        d_Q, d_K, B, T, n_q_heads, n_kv_heads, head_dim, d_sincos);
}

// ================================================================
// GQA RMSNorm + Causal Attention + Gate + Output Projection
// 
// This implements the full GQA attention step (CPU wubu_gqa_forward steps 3-6)
// on GPU using cuBLAS for matmuls and custom kernels for norms/attention.
//
// Steps:
//   1. Copy Q from Q_full (first q_dim) and RMSNorm with Q_norm_w
//   2. RMSNorm K with K_norm_w  
//   3. GQA attention: causal dot-product, softmax, weighted sum of V
//   4. Gate: sigmoid on gate portion of Q_full, multiply into attention out
//   5. Output projection: attn_out @ output_w
// ================================================================

// RMSNorm kernel with custom shape: [B, T*n_heads, head_dim]
__global__ void gqa_rms_norm_kernel(const float *x, const float *weight,
                                     float *out, int B, int T_nheads,
                                     int head_dim, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T_nheads) return;
    
    const float *inp = x + idx * head_dim;
    float *oup = out + idx * head_dim;
    
    // Double precision sum-of-squares for numerical stability
    double sum_sq = 0.0;
    for (int i = 0; i < head_dim; i++) sum_sq += (double)inp[i] * (double)inp[i];
    float rms = sqrtf((float)(sum_sq / head_dim) + eps);
    float scale = 1.0f / rms;
    for (int i = 0; i < head_dim; i++) oup[i] = inp[i] * scale * weight[i];
}

// Simple causal attention kernel: 1 thread per block (no shared mem issues)
// Each thread computes full dot-product, softmax, and weighted sum for one (b, t_q, h_q)
__global__ void causal_attn_simple_kernel(const float *Q, const float *K, const float *V,
                                           float *out, int B, int T,
                                           int n_q_heads, int n_kv_heads, int head_dim,
                                           float scale) {
    int idx = blockIdx.x;
    int h_q = idx % n_q_heads;
    int t_q = (idx / n_q_heads) % T;
    int b = idx / (T * n_q_heads);
    int h_kv = h_q / (n_q_heads / n_kv_heads);
    
    const float *q_vec = Q + ((b * T + t_q) * n_q_heads + h_q) * head_dim;
    float *out_vec = out + ((b * T + t_q) * n_q_heads + h_q) * head_dim;
    
    // Compute causal scores
    float attn_w[4096];
    float max_score = -1e30f;
    for (int t_k = 0; t_k <= t_q; t_k++) {
        const float *k_vec = K + ((b * T + t_k) * n_kv_heads + h_kv) * head_dim;
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) score += q_vec[i] * k_vec[i];
        score *= scale;
        attn_w[t_k] = score;
        if (score > max_score) max_score = score;
    }
    
    // Softmax
    float sum_exp = 0.0f;
    for (int t_k = 0; t_k <= t_q; t_k++) {
        attn_w[t_k] = expf(attn_w[t_k] - max_score);
        sum_exp += attn_w[t_k];
    }
    if (sum_exp > 0.0f) {
        float inv_sum = 1.0f / sum_exp;
        for (int t_k = 0; t_k <= t_q; t_k++) attn_w[t_k] *= inv_sum;
    }
    
    // Zero and accumulate
    for (int i = 0; i < head_dim; i++) out_vec[i] = 0.0f;
    for (int t_k = 0; t_k <= t_q; t_k++) {
        const float *v_vec = V + ((b * T + t_k) * n_kv_heads + h_kv) * head_dim;
        float a = attn_w[t_k];
        for (int i = 0; i < head_dim; i++) out_vec[i] += a * v_vec[i];
    }
}

// Sigmoid and multiply kernel: out[i] = x[i] * sigmoid(gate[i])
__global__ void gate_mul_kernel(float *x, const float *gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = gate[i];
    float s;
    if (v < -80.0f) s = 0.0f;
    else if (v > 80.0f) s = 1.0f;
    else s = 1.0f / (1.0f + expf(-v));
    x[i] *= s;
}

// Copy Q from fused Q+gate buffer: dest[s*qdim + j] = src[s*qdim*2 + j]
__global__ void copy_q_from_fused_kernel(float *dst, const float *src,
                                          int N, int qdim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * qdim) return;
    int s = i / qdim;
    int j = i % qdim;
    dst[i] = src[s * qdim * 2 + j];
}

// Copy gate from fused Q+gate buffer: dest[s*qdim + j] = src[s*qdim*2 + qdim + j]
__global__ void copy_gate_from_fused_kernel(float *dst, const float *src,
                                             int N, int qdim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * qdim) return;
    int s = i / qdim;
    int j = i % qdim;
    dst[i] = src[s * qdim * 2 + qdim + j];
}

// Gate multiply: x[s*qdim + j] *= sigmoid(Q_full[s*qdim*2 + qdim + j])
// Reads gate from fused Q+gate layout
__global__ void gate_mul_fused_kernel(float *x, const float *Q_full,
                                       int N, int qdim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * qdim) return;
    int s = i / qdim;
    int j = i % qdim;
    float v = Q_full[s * qdim * 2 + qdim + j];
    float sg;
    if (v < -80.0f) sg = 0.0f;
    else if (v > 80.0f) sg = 1.0f;
    else sg = 1.0f / (1.0f + expf(-v));
    x[i] *= sg;
}

void wubu_cuda_gqa_forward(cublasHandle_t handle, cudaStream_t stream,
    int B, int T,
    const float *d_Q_full,      // [N, q_dim*2] — first q_dim is Q, rest is gate
    const float *d_K,           // [N, kv_dim]
    const float *d_V,           // [N, kv_dim]
    const float *d_Q_norm_w,    // [HEAD_DIM]
    const float *d_K_norm_w,    // [HEAD_DIM]
    const float *d_output_w,    // [q_dim, D_MODEL]
    float *d_output,            // [N, D_MODEL] — final output
    float *d_scratch,           // [N, q_dim] — temp for attention out
    const float *d_sincos) {    // [T, ROTARY_DIM] — RoPE sin/cos, or NULL to skip
    const int N = B * T;
    const int q_dim = GQA_Q_HEADS * GQA_HEAD_DIM;   // 4096
    const int head_dim = GQA_HEAD_DIM;                   // 256
    
    // Step 1: Copy Q from d_Q_full (strided) to d_scratch (contiguous), then RMSNorm
    int block = 256;
    int grid_copy = (N * q_dim + block - 1) / block;
    copy_q_from_fused_kernel<<<grid_copy, block, 0, stream>>>(
        d_scratch, d_Q_full, N, q_dim);
    
    // RMSNorm Q: d_scratch now has [N, q_dim] = [N, Q_HEADS * HEAD_DIM]
    int grid_norm_q = (B * T * GQA_Q_HEADS + block - 1) / block;
    gqa_rms_norm_kernel<<<grid_norm_q, block, 0, stream>>>(
        d_scratch, d_Q_norm_w, d_scratch, B, T * GQA_Q_HEADS, head_dim, 1e-6f);
    
    // Step 2: RMSNorm K (in-place on d_K)
    int grid_norm_k = (B * T * GQA_KV_HEADS + block - 1) / block;
    gqa_rms_norm_kernel<<<grid_norm_k, block, 0, stream>>>(
        d_K, d_K_norm_w, (float*)d_K, B, T * GQA_KV_HEADS, head_dim, 1e-6f);
    
    // Step 2.5: Apply RoPE to Q and K (if sin/cos table provided)
    if (d_sincos != NULL) {
        wubu_cuda_apply_rotary_to_qk((float*)d_scratch, (float*)d_K,
            B, T, GQA_Q_HEADS, GQA_KV_HEADS, head_dim, d_sincos, stream);
    }

    // Step 3: Causal attention (1 thread per head)
    int n_blocks = B * T * GQA_Q_HEADS;
    causal_attn_simple_kernel<<<n_blocks, 1, 0, stream>>>(  // 1 thread per block
        d_scratch, d_K, d_V, d_scratch, B, T,
        GQA_Q_HEADS, GQA_KV_HEADS, head_dim,
        1.0f / sqrtf((float)head_dim));
    
    // Step 4: Gate — sigmoid then multiply
    int grid_gate = (N * q_dim + block - 1) / block;
    gate_mul_fused_kernel<<<grid_gate, block, 0, stream>>>(
        d_scratch, d_Q_full, N, q_dim);
    
    // Step 5: Output projection
    wubu_cuda_matmul(handle, d_scratch, N, q_dim, d_output_w, D_MODEL, d_output, 1.0f, 0.0f);
}

// ================================================================
// GQA attention only — for save variant (backward)
// Assumes Q and K are already RMSNorm'd.
// ================================================================
void wubu_cuda_gqa_attention_only(cublasHandle_t handle, cudaStream_t stream,
    int B, int T,
    const float *d_Q,       // [N, q_dim] post-RMSNorm
    const float *d_K,       // [N, kv_dim] post-RMSNorm
    const float *d_V,       // [N, kv_dim]
    float *d_output,        // [N, q_dim] attention output (pre-gate)
    int n_q_heads, int n_kv_heads, int head_dim) {
    (void)handle;
    int n_blocks = B * T * n_q_heads;
    causal_attn_simple_kernel<<<n_blocks, 1, 0, stream>>>(
        d_Q, d_K, d_V, d_output, B, T,
        n_q_heads, n_kv_heads, head_dim,
        1.0f / sqrtf((float)head_dim));
}

// ================================================================
// GQA gate multiply — for save variant (backward)
// x *= sigmoid(gate_part_of_Q_full)
// ================================================================
void wubu_cuda_gqa_gate(float *d_x, const float *d_Q_full,
    int N, int q_dim, cudaStream_t stream) {
    int block = 256;
    int grid = (N * q_dim + block - 1) / block;
    gate_mul_fused_kernel<<<grid, block, 0, stream>>>(d_x, d_Q_full, N, q_dim);
}

// ================================================================
// Hyperbolic (Poincaré ball) CUDA kernels
// ================================================================

// Compute norm of each vector in a batch: out[i] = sqrt(sum_k in[i*d+k]^2)
__global__ void norm_kernel(const float *in, float *out, int n_vecs, int dim) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int vec = blockIdx.x;
    if (vec >= n_vecs) return;
    const float *v = in + vec * dim;
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x)
        sum += v[i] * v[i];
    shared[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[vec] = sqrtf(shared[0]);
}

// exp_map: out[i] = tanh(||v||/R) * v[i] / ||v||  (for ||v|| > 1e-8)
__global__ void exp_map_kernel(const float *v, const float *norms, float R,
                                float *out, int n_vecs, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vecs * dim) return;
    int vec = idx / dim;
    int i = idx % dim;
    float n = norms[vec];
    if (n < 1e-8f) { out[idx] = v[idx]; return; }
    out[idx] = R * tanhf(n / R) * v[idx] / n;
}

// log_map: out[i] = artanh(||v||/R) * v[i] / ||v|| * R  (for ||v|| > 1e-8)
__global__ void log_map_kernel(const float *v, const float *norms, float R,
                                float *out, int n_vecs, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vecs * dim) return;
    int vec = idx / dim;
    int i = idx % dim;
    float n = norms[vec];
    if (n < 1e-8f) { out[idx] = v[idx]; return; }
    float ratio = n / R;
    if (ratio >= 1.0f) ratio = 0.99f; // clamp
    out[idx] = (float)(atanh((double)ratio) / (double)n) * v[idx] * R;
}

// Möbius scalar multiplication: out = r ⊗ v
// out[i] = tanh(r * artanh(||v||/R)) * v[i] / ||v|| * R
__global__ void mobius_scalar_mul_kernel(const float *v, const float *norms, float r, float R,
                                          float *out, int n_vecs, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vecs * dim) return;
    int vec = idx / dim;
    int i = idx % dim;
    float n = norms[vec];
    if (n < 1e-8f || fabsf(r) < 1e-8f) { out[idx] = v[idx]; return; }
    float ratio = n / R;
    if (ratio >= 1.0f) ratio = 0.99f;
    float artanh_n = (float)atanh((double)ratio);
    float factor = tanhf(r * artanh_n) * R / n;
    out[idx] = v[idx] * factor;
}

// Möbius addition: out = x ⊕ y
// z = ((1+2⟨x,y⟩+||y||²)x + (1-||x||²)y) / (1+2⟨x,y⟩+||x||²||y||²)
__global__ void mobius_add_kernel(const float *x, const float *y,
                                   const float *nx2, const float *ny2,
                                   float *out, int n_vecs, int dim) {
    extern __shared__ float shared_dot[];
    int tid = threadIdx.x;
    int vec = blockIdx.x;
    if (vec >= n_vecs) return;
    const float *xv = x + vec * dim;
    const float *yv = y + vec * dim;
    float *ov = out + vec * dim;
    
    // Compute dot product ⟨x,y⟩
    float dot_val = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x)
        dot_val += xv[i] * yv[i];
    shared_dot[tid] = dot_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared_dot[tid] += shared_dot[tid + s];
        __syncthreads();
    }
    float xy = shared_dot[0];
    __syncthreads();
    
    float n_x2 = nx2[vec];
    float n_y2 = ny2[vec];
    float denom = 1.0f + 2.0f * xy + n_x2 * n_y2;
    float inv_denom = 1.0f / (denom + 1e-30f);
    float coeff_x = (1.0f + 2.0f * xy + n_y2) * inv_denom;
    float coeff_y = (1.0f - n_x2) * inv_denom;
    
    for (int i = tid; i < dim; i += blockDim.x)
        ov[i] = coeff_x * xv[i] + coeff_y * yv[i];
}

// ================================================================
// Host wrappers for hyperbolic kernels (declare in cuda_kernels.h)
// ================================================================
void wubu_cuda_norm(const float *d_in, float *d_out, int n_vecs, int dim, cudaStream_t stream) {
    const int block = 256;
    const int shmem = block * sizeof(float);
    norm_kernel<<<n_vecs, block, shmem, stream>>>(d_in, d_out, n_vecs, dim);
}

void wubu_cuda_exp_map(const float *d_v, const float *d_norms, float R,
                        float *d_out, int n_vecs, int dim, cudaStream_t stream) {
    const int block = 256;
    const int grid = (n_vecs * dim + block - 1) / block;
    exp_map_kernel<<<grid, block, 0, stream>>>(d_v, d_norms, R, d_out, n_vecs, dim);
}

void wubu_cuda_log_map(const float *d_v, const float *d_norms, float R,
                        float *d_out, int n_vecs, int dim, cudaStream_t stream) {
    const int block = 256;
    const int grid = (n_vecs * dim + block - 1) / block;
    log_map_kernel<<<grid, block, 0, stream>>>(d_v, d_norms, R, d_out, n_vecs, dim);
}

void wubu_cuda_mobius_scalar_mul(const float *d_v, const float *d_norms,
                                  float r, float R, float *d_out,
                                  int n_vecs, int dim, cudaStream_t stream) {
    const int block = 256;
    const int grid = (n_vecs * dim + block - 1) / block;
    mobius_scalar_mul_kernel<<<grid, block, 0, stream>>>(d_v, d_norms, r, R, d_out, n_vecs, dim);
}

void wubu_cuda_mobius_add(const float *d_x, const float *d_y,
                           const float *d_nx2, const float *d_ny2,
                           float *d_out, int n_vecs, int dim, cudaStream_t stream) {
    const int block = 256;
    const int shmem = block * sizeof(float);
    mobius_add_kernel<<<n_vecs, block, shmem, stream>>>(d_x, d_y, d_nx2, d_ny2, d_out, n_vecs, dim);
}

// ================================================================
// Poincaré recurrence kernel: replaces Euclidean step 9
// Processes all (B, T, V_HEADS) in parallel.
// Each block handles one head for one token in one batch item.
// Processes rows one-at-a-time to stay within 48KB shared memory limit.
// ================================================================
__global__ void poincare_recurrence_kernel(
    float *d_h_states,      // [B, SSM_V_HEADS, D_STATE, D_STATE] mutable
    const float *d_q_norm,  // [N, SSM_K_HEADS, D_STATE]
    const float *d_k_norm,  // [N, SSM_K_HEADS, D_STATE]
    const float *d_v_conv,  // [N, SSM_V_HEADS, D_STATE]
    const float *d_gate,    // [N, DT_RANK]
    const float *d_beta,    // [N, DT_RANK]
    float *d_delta_out,     // [N, SSM_V_HEADS, D_STATE]
    int B, int T, float R,
    float *d_states_t)      // [B, T+1, SSM_V_HEADS, D_STATE, D_STATE] — NULL ok
{
    int b = blockIdx.x;
    int t = blockIdx.y;
    int vh = blockIdx.z;
    int s = b * T + t;
    int kh = vh / 2;
    
    if (b >= B || t >= T || vh >= SSM_V_HEADS) return;
    
    float bg = d_beta[s * DT_RANK + kh];
    float gg = tgt_safe_expf(d_gate[s * DT_RANK + kh]);
    
    int tid = threadIdx.x;
    if (tid >= SSM_D_STATE) return;
    
    // Shared memory for per-row data (512 bytes per row)
    __shared__ float sh_k[SSM_D_STATE];
    __shared__ float sh_v[SSM_D_STATE];
    __shared__ float sh_q[SSM_D_STATE];
    __shared__ float sh_v_tan[SSM_D_STATE];
    
    // Load K, V, Q from global
    sh_k[tid] = d_k_norm[(s * SSM_K_HEADS + kh) * SSM_D_STATE + tid];
    sh_v[tid] = d_v_conv[(s * SSM_V_HEADS + vh) * SSM_D_STATE + tid];
    sh_q[tid] = d_q_norm[(s * SSM_K_HEADS + kh) * SSM_D_STATE + tid];
    __syncthreads();
    
    // Pre-compute log_map(v) — same for all rows
    {
        __shared__ float vn2[SSM_D_STATE];
        vn2[tid] = sh_v[tid] * sh_v[tid];
        __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
            if (tid < s2) vn2[tid] += vn2[tid + s2];
            __syncthreads();
        }
        float nv = sqrtf(vn2[0]);
        __syncthreads();
        if (nv < 1e-8f) {
            sh_v_tan[tid] = sh_v[tid];
        } else {
            float ratio = nv / R;
            if (ratio >= 1.0f) ratio = 0.99f;
            sh_v_tan[tid] = (float)(atanh((double)ratio) / (double)nv) * sh_v[tid] * R;
        }
    }
    __syncthreads();
    
    // Pointer to this head's state matrix in global memory
    float *h_state = d_h_states + (b * SSM_V_HEADS + vh) * SSM_D_STATE * SSM_D_STATE;
    float out_val = 0.0f;
    
    // Process each row of h_state one at a time
    for (int row = 0; row < SSM_D_STATE; row++) {
        float *h_row = h_state + row * SSM_D_STATE;
        
        // Save state to trajectory: h_t = h_state before modification (thread 0 only)
        if (d_states_t && tid == 0 && row == 0) {
            float *traj = d_states_t + ((s * SSM_V_HEADS + vh) * (T+1) + t) * SSM_D_STATE * SSM_D_STATE;
            for (int r = 0; r < SSM_D_STATE; r++)
                for (int c = 0; c < SSM_D_STATE; c++)
                    traj[r * SSM_D_STATE + c] = h_state[r * SSM_D_STATE + c];
        }
        
        // Step 9a: Möbius scalar decay
        float h_val = h_row[tid];
        __shared__ float rn2[SSM_D_STATE];
        rn2[tid] = h_val * h_val;
        __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
            if (tid < s2) rn2[tid] += rn2[tid + s2];
            __syncthreads();
        }
        float n_h = sqrtf(rn2[0]);
        __syncthreads();
        
        if (n_h > 1e-8f && fabsf(gg) > 1e-8f) {
            float ratio = n_h / R;
            if (ratio >= 1.0f) ratio = 0.99f;
            float art = (float)atanh((double)ratio);
            float factor = tanhf(gg * art) * R / n_h;
            h_val *= factor;
            h_row[tid] = h_val; // write decayed value back immediately
        }
        __syncthreads();
        
        // Step 9b: hk_tan = dot(log_map(h_row), k)
        float log_h;
        if (n_h < 1e-8f) {
            log_h = h_val;
        } else {
            float ratio = n_h / R;
            if (ratio >= 1.0f) ratio = 0.99f;
            log_h = (float)(atanh((double)ratio) / (double)n_h) * h_val * R;
        }
        
        __shared__ float hk_s[SSM_D_STATE];
        hk_s[tid] = log_h * sh_k[tid];
        __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
            if (tid < s2) hk_s[tid] += hk_s[tid + s2];
            __syncthreads();
        }
        float hk = hk_s[0];
        __syncthreads();
        
        // Steps 9c-d: diff_tan = v_tan - hk, update = k * diff * bg
        float diff = sh_v_tan[tid] - hk;
        float upd_tan = sh_k[tid] * diff * bg;
        
        // Step 9e: exp_map(update)
        __shared__ float un2[SSM_D_STATE];
        un2[tid] = upd_tan * upd_tan;
        __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
            if (tid < s2) un2[tid] += un2[tid + s2];
            __syncthreads();
        }
        float nu = sqrtf(un2[0]);
        __syncthreads();
        
        float upd_ball;
        if (nu < 1e-8f) {
            upd_ball = upd_tan;
        } else {
            upd_ball = R * tanhf(nu / R) * upd_tan / nu;
        }
        
        // Step 9f: Möbius add: h_row = h_row ⊕ upd_ball
        float n_h2 = rn2[0]; // already computed above
        __shared__ float ub2_s[SSM_D_STATE];
        ub2_s[tid] = upd_ball * upd_ball;
        __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
            if (tid < s2) ub2_s[tid] += ub2_s[tid + s2];
            __syncthreads();
        }
        float n_ub2 = ub2_s[0];
        __syncthreads();
        
        __shared__ float dot_s[SSM_D_STATE];
        dot_s[tid] = h_val * upd_ball;
        __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
            if (tid < s2) dot_s[tid] += dot_s[tid + s2];
            __syncthreads();
        }
        float xy = dot_s[0];
        __syncthreads();
        
        float denom = 1.0f + 2.0f * xy + n_h2 * n_ub2;
        float inv_denom = 1.0f / (denom + 1e-30f);
        float coeff_x = (1.0f + 2.0f * xy + n_ub2) * inv_denom;
        float coeff_y = (1.0f - n_h2) * inv_denom;
        
        // Save state to trajectory buffer before update
        // (captures h_state at timestep t, before this timestep's modification)
        // Trajectory save happens before h_row modification for timestep t
        // d_states_t[b][t][vh][row][col] = h_state... but we need to save
        // the state BEFORE this iteration's decay+update.
        // The state at entry to each timestep is what we need for backward.
        
        h_row[tid] = coeff_x * h_val + coeff_y * upd_ball;
        __syncthreads();
        
        // Accumulate output contribution: out[tid] += h_row[tid] * q_row[tid]
        // But this is wrong — it's actually sum_row h_row[tid] * sh_q[row]
        // We need the matvec h @ q for column tid
        // out[tid] = sum_row h_state[row][tid] * q[row]
        // Actually: q is indexed by 'row', not by 'tid'. 
        // Thread tid handles column 'tid'. We multiply by q[row] and accumulate.
        // This requires knowing all rows' values for column tid.
    }
    
    // Step 9g: output = h @ q  (Euclidean matvec)
    // Each thread computes: out[tid] = sum_row h_state[row][tid] * sh_q[row]
    float result = 0.0f;
    for (int row = 0; row < SSM_D_STATE; row++) {
        result += h_state[row * SSM_D_STATE + tid] * sh_q[row];
    }
    d_delta_out[(s * SSM_V_HEADS + vh) * SSM_D_STATE + tid] = result;
}

// Host wrapper for Poincaré recurrence
void wubu_cuda_poincare_recurrence(cublasHandle_t handle, cudaStream_t stream,
    int B, int T, float R,
    float *d_h_states,
    const float *d_q_norm, const float *d_k_norm, const float *d_v_conv,
    const float *d_gate, const float *d_beta,
    float *d_delta_out,
    float *d_states_t)   // optional trajectory buffer
{
    dim3 grid(B, T, SSM_V_HEADS);
    poincare_recurrence_kernel<<<grid, SSM_D_STATE, 0, stream>>>(
        d_h_states, d_q_norm, d_k_norm, d_v_conv, d_gate, d_beta,
        d_delta_out, B, T, R, d_states_t);
}

// ================================================================
// SSM SCALAR PARALLEL ASSOCIATIVE SCAN
//
// Implements: h[t][i] = A[t] * h[t-1][i] + B[t] * v[t][i]
//
// Each dimension i is independent. We use a true parallel scan
// across T time steps for each dimension simultaneously.
//
// Grid: (B_ * d) blocks, each block = T threads
// Each block handles one (batch, dim) pair.
// Uses shared memory tree (Blelloch up-sweep) for inclusive scan.
//
// Associative operator: (a,b) ∘ (c,d) = (a*c, a*d + b)
// representing: h -> a*h + b
// ================================================================

__global__ void ssm_scalar_scan_kernel(
    const float *A,            // [B_, T]
    const float *B,            // [B_, T]
    const float *v,            // [B_, T, d]
    float *h,                  // [B_, d] — initial (in), final (out)
    float *delta_out,          // [B_, T, d]
    int B_, int T, int d) {

    // blockIdx.x = batch * d + dim
    int batch = blockIdx.x / d;
    int dim = blockIdx.x % d;
    if (batch >= B_ || dim >= d) return;

    int t = threadIdx.x;  // 0..T-1
    if (t >= T) return;

    extern __shared__ float sh_pair[];  // [T, 2] — but we need 2x for double buffer
    // We'll use sh_pair[0..T*2-1] as buffer A, sh_pair[T*2..T*4-1] as buffer B
    float *bufA = sh_pair;
    float *bufB = sh_pair + T * 2;

    // Compute initial pair for this (batch, dim, t)
    float a_val = A[batch * T + t];
    float b_val = B[batch * T + t] * v[((batch * T) + t) * d + dim];

    // Store in buffer A
    bufA[t * 2 + 0] = a_val;
    bufA[t * 2 + 1] = b_val;
    __syncthreads();

    // Hillis-Steele parallel prefix scan (correct double-buffered version)
    // This gives inclusive scan: pairs[t] = pair[0]∘...∘pair[t]
    for (int stride = 1; stride < T; stride <<= 1) {
        if (t >= stride) {
            // Compose: apply element at (t-stride) FIRST, then element at t
            // compose(prev, curr) = (prev.a*curr.a, prev.a*curr.b + prev.b)
            // This means: first apply curr, then apply prev — WRONG order.
            // We need to apply (t-stride) first, then t.
            // So: compose(curr, prev) = (curr.a*prev.a, curr.a*prev.b + curr.b)
            float pa = bufA[(t - stride) * 2 + 0];  // prev.a
            float pb = bufA[(t - stride) * 2 + 1];  // prev.b
            float ca = bufA[t * 2 + 0];              // curr.a
            float cb = bufA[t * 2 + 1];              // curr.b
            bufB[t * 2 + 0] = ca * pa;               // curr.a * prev.a
            bufB[t * 2 + 1] = ca * pb + cb;          // curr.a * prev.b + curr.b
        } else {
            bufB[t * 2 + 0] = bufA[t * 2 + 0];
            bufB[t * 2 + 1] = bufA[t * 2 + 1];
        }
        __syncthreads();
        // Swap buffers: copy B back to A for next iteration
        bufA[t * 2 + 0] = bufB[t * 2 + 0];
        bufA[t * 2 + 1] = bufB[t * 2 + 1];
        __syncthreads();
    }

    // After double-buffered loop, bufA has the inclusive scan result
    // pairs[t] = (Π(A[0..t]), h[t] assuming h[0]=0)
    // So h[t] = Π(A[0..t]) * h0 + (inclusive b-scan)

    float prefix_a = bufA[t * 2 + 0];
    float prefix_b = bufA[t * 2 + 1];

    // Apply initial state: h[t] = prefix_a * h[0] + prefix_b
    float h0 = h[batch * d + dim];
    float out_val = prefix_a * h0 + prefix_b;

    delta_out[((batch * T) + t) * d + dim] = out_val;

    if (t == T - 1) {
        h[batch * d + dim] = out_val;
    }
}

void wubu_cuda_ssm_scalar_scan(int B_, int T, int d,
    const float *d_A,
    const float *d_B,
    const float *d_v,
    float *d_h,
    float *d_delta_out,
    cudaStream_t stream) {

    // Each block handles one (batch, dim) pair with T threads
    int grid = B_ * d;
    int block = T;
    if (block > 1024) block = 1024;  // safety

    size_t shmem = T * 2 * 2 * sizeof(float);  // double buffer: [T, 2] * 2
    ssm_scalar_scan_kernel<<<grid, block, shmem, stream>>>(
        d_A, d_B, d_v, d_h, d_delta_out, B_, T, d);
}

// ================================================================
// MoE DISPATCH KERNEL
//
// Groups tokens by expert assignment, does one batched matmul per
// expert using cuBLAS.
//
// Algorithm:
//   1. Build histogram of tokens per expert (from assignments)
//   2. Exclusive prefix sum over histogram → expert token offsets
//   3. Scatter tokens into expert-grouped buffer
//   4. For each expert with >0 tokens:
//      a. gate = tokens @ gate_weight  [n_tok, D_FF]
//      b. up   = tokens @ up_weight    [n_tok, D_FF]
//      c. act = SiLU(gate) * up
//      d. down = act @ down_weight     [n_tok, D_MODEL]
//      e. Scatter back, weighted by routing weights
//
// Shared expert is handled separately (less work, all tokens).
// ================================================================

// Kernel: histogram count of tokens per expert
// assignments: [B*T, N_ACTIVE_EXPTS] — expert indices
// weights: [B*T, N_ACTIVE_EXPTS] — routing weights (skip near-zero)
// histogram: [N_EXPERTS] — output counts (must be zeroed)
__global__ void moe_histogram_kernel(const int *assignments,
                                     const float *weights,
                                     int *histogram,
                                     int N, int topk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * topk) return;
    float w = weights[idx];
    if (w < 1e-30f) return;
    int e = assignments[idx];
    if (e >= 0 && e < N_EXPERTS) {
        atomicAdd(&histogram[e], 1);
    }
}

// Kernel: scatter tokens into expert-grouped buffer + weight buffer + token_map
// assignments: [B*T, N_ACTIVE_EXPTS] — expert indices
// weights: [B*T, N_ACTIVE_EXPTS] — routing weights
// x: [B*T, D_MODEL] — input tokens
// expert_offsets: [N_EXPERTS] — exclusive prefix sum of histogram
// permuted_x: [total_assigned, D_MODEL] — token buffer grouped by expert
// permuted_w: [total_assigned] — routing weights in same order
// token_map: [total_assigned] — original token index for each slot
// expert_token_idx: running counter (per-expert) — must be zeroed
__global__ void moe_scatter_kernel(const float *x,
                                   const int *assignments,
                                   const float *weights,
                                   const int *expert_offsets,
                                   float *permuted_x,
                                   float *permuted_w,
                                   int *token_map,
                                   int *expert_token_idx,
                                   int N, int topk, int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * topk) return;
    int token = idx / topk;
    int k = idx % topk;
    int e = assignments[idx];
    if (e < 0 || e >= N_EXPERTS) return;
    float w = weights[idx];
    if (w < 1e-30f) return;

    // Atomically claim the next slot for this expert
    int slot = atomicAdd(&expert_token_idx[e], 1);
    int write_pos = expert_offsets[e] + slot;

    // Copy token vector to permuted buffer
    const float *x_src = x + token * d_model;
    float *x_dst = permuted_x + write_pos * d_model;
    for (int i = 0; i < d_model; i++) {
        x_dst[i] = x_src[i];
    }

    // Store weight and original token index
    permuted_w[write_pos] = w;
    token_map[write_pos] = token;
}

// Kernel: scatter back expert outputs to original token positions
// Adds weighted expert output to the token's position in the output buffer.
// output: [B*T, D_MODEL] — accumulated output (zero-init)
// permuted_out: [total_assigned, D_MODEL] — expert outputs grouped by expert
// token_map: [total_assigned] — original token index for each slot
// weights: [total_assigned] — routing weight for each assignment
__global__ void moe_scatter_back_kernel(float *output,
                                        const float *permuted_out,
                                        const int *token_map,
                                        const float *weights,
                                        int total_assigned,
                                        int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_assigned * d_model) return;
    int slot = idx / d_model;
    int dim = idx % d_model;
    int token = token_map[slot];
    float w = weights[slot];
    atomicAdd(&output[token * d_model + dim], w * permuted_out[idx]);
}

// Element-wise multiply kernel: y[i] = x[i] * z[i]
// For act = silu(gate) * up
__global__ void elemul_kernel(float *y, const float *x, const float *z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i] * z[i];
}

// Element-wise add kernel: y[i] += x[i]
__global__ void elemadd_kernel(float *y, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] += x[i];
}

// Query scratch size for MoE dispatch
size_t wubu_cuda_moe_dispatch_query_scratch(int B, int T) {
    int N = B * T;
    int max_assigned = N * N_ACTIVE_EXPTS;  // upper bound
    // histogram: N_EXPERTS ints
    // expert_offsets: N_EXPERTS ints
    // expert_token_idx: N_EXPERTS ints
    // permuted_x: max_assigned * D_MODEL floats
    // permuted_w: max_assigned floats
    // token_map: max_assigned ints
    // gate_out: max_assigned * D_FF floats (temp for matmul)
    // up_out: max_assigned * D_FF floats
    // act_out: max_assigned * D_FF floats
    // permuted_out: max_assigned * D_MODEL floats
    size_t total = 0;
    total += N_EXPERTS * 3 * sizeof(int);          // histogram + offsets + idx
    total += max_assigned * D_MODEL * sizeof(float); // permuted_x
    total += max_assigned * sizeof(float);           // permuted_w
    total += max_assigned * sizeof(int);             // token_map
    total += max_assigned * D_FF * 3 * sizeof(float); // gate + up + act
    total += max_assigned * D_MODEL * sizeof(float); // permuted_out
    total += 1024 * 1024;  // padding
    return total;
}

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
    float *d_scratch) {

    int N = B * T;
    int max_assigned = N * N_ACTIVE_EXPTS;

    // Scratch layout (offsets in bytes): using char* arithmetic
    size_t offset = 0;
    int *d_histogram     = (int*)((char*)d_scratch + offset); offset += N_EXPERTS * sizeof(int);
    int *d_expert_offsets= (int*)((char*)d_scratch + offset); offset += N_EXPERTS * sizeof(int);
    int *d_expert_idx    = (int*)((char*)d_scratch + offset); offset += N_EXPERTS * sizeof(int);
    float *d_permuted_x  = (float*)((char*)d_scratch + offset); offset += max_assigned * D_MODEL * sizeof(float);
    float *d_permuted_w  = (float*)((char*)d_scratch + offset); offset += max_assigned * sizeof(float);
    int *d_token_map     = (int*)((char*)d_scratch + offset); offset += max_assigned * sizeof(int);
    float *d_gate_out    = (float*)((char*)d_scratch + offset); offset += max_assigned * D_FF * sizeof(float);
    float *d_up_out      = (float*)((char*)d_scratch + offset); offset += max_assigned * D_FF * sizeof(float);
    float *d_act_out     = (float*)((char*)d_scratch + offset); offset += max_assigned * D_FF * sizeof(float);
    float *d_permuted_out= (float*)((char*)d_scratch + offset); offset += max_assigned * D_MODEL * sizeof(float);
    // Remaining scratch is padding

    int block = 256;

    // Step 0: Zero histograms and counter
    cudaMemsetAsync(d_histogram, 0, N_EXPERTS * sizeof(int), stream);
    cudaMemsetAsync(d_expert_idx, 0, N_EXPERTS * sizeof(int), stream);

    // Step 1: Compute histogram of tokens per expert
    int grid_hist = (N * N_ACTIVE_EXPTS + block - 1) / block;
    moe_histogram_kernel<<<grid_hist, block, 0, stream>>>(
        d_assignments, d_weights, d_histogram, N, N_ACTIVE_EXPTS);

    // Step 2: Exclusive prefix sum over histogram → expert_offsets
    // Simple sequential kernel since N_EXPERTS=256 is small
    // We'll do it on device: first a kernel copies, then does prefix
    // Actually let's do a simple kernel with 1 block, 256 threads
    {
        // Copy histogram to offsets first
        int grid_cp = (N_EXPERTS + block - 1) / block;
        // We'll do prefix sum in a separate kernel with 1 block
        // For simplicity, use cudaMemcpy + sequential CPU for 256 ints
        // This is fast enough for N_EXPERTS=256
        int hist_host[N_EXPERTS];
        cudaMemcpyAsync(hist_host, d_histogram, N_EXPERTS * sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        int offsets[N_EXPERTS + 1];
        offsets[0] = 0;
        for (int i = 0; i < N_EXPERTS; i++) {
            offsets[i + 1] = offsets[i] + hist_host[i];
        }
        int total_assigned = offsets[N_EXPERTS];
        cudaMemcpyAsync(d_expert_offsets, offsets, N_EXPERTS * sizeof(int),
                        cudaMemcpyHostToDevice, stream);

        // Step 3: Scatter tokens into permuted buffer
        moe_scatter_kernel<<<grid_hist, block, 0, stream>>>(
            d_x, d_assignments, d_weights, d_expert_offsets,
            d_permuted_x, d_permuted_w, d_token_map, d_expert_idx,
            N, N_ACTIVE_EXPTS, D_MODEL);

        // Step 4: For each expert with >0 tokens, do matmuls
        // We iterate over experts sequentially on CPU, launching cuBLAS
        for (int e = 0; e < N_EXPERTS; e++) {
            int n_tok = hist_host[e];
            if (n_tok == 0) continue;

            const float *tok_buf = d_permuted_x + offsets[e] * D_MODEL;
            const float *w_gate = d_gate_exps + (int64_t)e * D_MODEL * D_FF;
            const float *w_up   = d_up_exps   + (int64_t)e * D_MODEL * D_FF;
            const float *w_down = d_down_exps  + (int64_t)e * D_FF * D_MODEL;

            float *gate_buf = d_gate_out + offsets[e] * D_FF;
            float *up_buf   = d_up_out   + offsets[e] * D_FF;
            float *act_buf  = d_act_out  + offsets[e] * D_FF;
            float *out_buf  = d_permuted_out + offsets[e] * D_MODEL;

            // gate = tokens @ gate_weight  [n_tok, D_MODEL] @ [D_MODEL, D_FF] -> [n_tok, D_FF]
            wubu_cuda_matmul(handle, tok_buf, n_tok, D_MODEL, w_gate, D_FF, gate_buf, 1.0f, 0.0f);

            // up   = tokens @ up_weight    [n_tok, D_MODEL] @ [D_MODEL, D_FF] -> [n_tok, D_FF]
            wubu_cuda_matmul(handle, tok_buf, n_tok, D_MODEL, w_up, D_FF, up_buf, 1.0f, 0.0f);

            // act = SiLU(gate) * up
            wubu_cuda_silu(n_tok * D_FF, gate_buf, act_buf, stream);
            // Element-wise multiply: act_buf = act_buf * up_buf
            {
                int b = 256;
                int g = (n_tok * D_FF + b - 1) / b;
                elemul_kernel<<<g, b, 0, stream>>>(act_buf, act_buf, up_buf, n_tok * D_FF);
            }

            // out = act @ down_weight  [n_tok, D_FF] @ [D_FF, D_MODEL] -> [n_tok, D_MODEL]
            wubu_cuda_matmul(handle, act_buf, n_tok, D_FF, w_down, D_MODEL, out_buf, 1.0f, 0.0f);
        }

        // Step 5: Scatter back weighted outputs to original token positions
        // total_assigned is already computed on host at line offsets[N_EXPERTS]
        int total_assigned_final = total_assigned;
        cudaMemsetAsync(d_output, 0, N * D_MODEL * sizeof(float), stream);
        if (total_assigned_final > 0) {
            int g_sb = (total_assigned_final * D_MODEL + 255) / 256;
            moe_scatter_back_kernel<<<g_sb, 256, 0, stream>>>(
                d_output, d_permuted_out, d_token_map, d_permuted_w,
                total_assigned_final, D_MODEL);
        }

        // Step 6: Shared expert contribution (all tokens)
        // Use d_gate_out (first N * D_FF entries) as temp for shared expert per-token results
        // Since D_FF >= SHARED_D_FF, this is safe: d_gate_out has N*D_FF entries.
        {
            float *shared_temp = d_gate_out;
            // gate: [N, D_MODEL] @ [D_MODEL, SHARED_D_FF] -> [N, SHARED_D_FF]
            wubu_cuda_matmul(handle, d_x, N, D_MODEL, d_gate_shexp, SHARED_D_FF, shared_temp, 1.0f, 0.0f);
            // up: [N, D_MODEL] @ [D_MODEL, SHARED_D_FF] -> [N, SHARED_D_FF] -> store in d_up_out region
            wubu_cuda_matmul(handle, d_x, N, D_MODEL, d_up_shexp, SHARED_D_FF, d_up_out, 1.0f, 0.0f);
            // SiLU gate -> store back to shared_temp
            wubu_cuda_silu(N * SHARED_D_FF, shared_temp, shared_temp, stream);
            // Multiply by up: shared_temp = SiLU(gate) * up
            {
                int b = 256;
                int g = (N * SHARED_D_FF + b - 1) / b;
                elemul_kernel<<<g, b, 0, stream>>>(shared_temp, shared_temp, d_up_out, N * SHARED_D_FF);
            }
            // Down proj: [N, SHARED_D_FF] @ [SHARED_D_FF, D_MODEL] -> [N, D_MODEL], add to output
            wubu_cuda_matmul(handle, shared_temp, N, SHARED_D_FF, d_down_shexp, D_MODEL, d_output, 1.0f, 1.0f);
        }
    }
}

// (elemul_kernel and elemadd_kernel are defined above)

// ================================================================
// GPU MoE helper kernels
// ================================================================

// SiLU multiply: y = SiLU(gate) * up = gate * sigmoid(gate) * up
__global__ void silu_mul_kernel(const float *gate, const float *up, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float s = (g < -80.0f) ? 0.0f : (g > 80.0f) ? 1.0f : 1.0f / (1.0f + expf(-g));
    y[i] = g * s * up[i];  // SwiGLU: SiLU(gate) * up
}

// Element-wise add: y[i] = x[i] + y[i]
__global__ void add_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] += x[i];
}

// ================================================================
// GPU MoE forward — single token against active experts
//
// x: [D_MODEL] input token
// gw/uw/dw: [N_EXPS, D_MODEL * D_FF] or [N_EXPS, D_FF * D_MODEL] dequantized FP32 on HOST
// eids: [N_ACTIVE] expert IDs to use
// ewgts: [N_ACTIVE] routing weights
// n_active: number of active experts (usually 8)
// out: [D_MODEL] output (GPU)
//
// Uploads dequantized expert weights to GPU, does cuBLAS SGEMM,
// sums weighted results.
// ================================================================
void wubu_cuda_moe_fwd_1tok(cublasHandle_t handle, cudaStream_t stream,
    const float *d_x,           // [D_MODEL] on GPU
    const float *const*gw,      // [N_ACTIVE][D_MODEL, D_FF] host dequantized
    const float *const*uw,      // [N_ACTIVE][D_MODEL, D_FF]
    const float *const*dw,      // [N_ACTIVE][D_FF, D_MODEL]
    const int *eids,            // [N_ACTIVE] (unused, for logging)
    const float *ewgts,         // [N_ACTIVE]
    int n_active,
    float *d_out,               // [D_MODEL] output (GPU)
    float *d_gate_w,            // [D_MODEL * D_FF] scratch: upload gate weight, then gate result
    float *d_up_w,              // [D_MODEL * D_FF] scratch: upload up weight, then up result
    float *d_silu,              // [D_FF] scratch: SiLU output
    float *d_dn_w,              // [D_FF * D_MODEL] scratch: upload down weight
    float *d_contrib)           // [D_MODEL] scratch: per-expert contribution
{
    const int dm = D_MODEL, df = D_FF;

    // Zero output
    { int block = 256; int grid = (dm + block - 1) / block;
      init_zero_kernel<<<grid, block, 0, stream>>>(d_out, dm); }

    for (int k = 0; k < n_active; k++) {
        if (ewgts[k] < 1e-30f) continue;

        // Upload gate weight [dm, df] → result [1, df]
        cudaMemcpyAsync(d_gate_w, gw[k], (size_t)dm * df * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_up_w, uw[k], (size_t)dm * df * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_dn_w, dw[k], (size_t)df * dm * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        // Gate proj: x [1,dm] @ gate_w [dm,df] → d_gate_w [1,df]
        wubu_cuda_matmul(handle, d_x, 1, dm, d_gate_w, df, d_gate_w, 1.0f, 0.0f);

        // Up proj: x [1,dm] @ up_w [dm,df] → d_up_w [1,df]
        wubu_cuda_matmul(handle, d_x, 1, dm, d_up_w, df, d_up_w, 1.0f, 0.0f);

        // SiLU: d_silu[i] = d_gate_w[i] * sigmoid(d_gate_w[i]) * d_up_w[i]
        { int block = 256; int grid = (df + block - 1) / block;
          silu_mul_kernel<<<grid, block, 0, stream>>>(d_gate_w, d_up_w, d_silu, df); }

        // Down proj: d_silu [1,df] @ dn_w [df,dm] → d_contrib [1,dm] * ewgts[k]
        wubu_cuda_matmul(handle, d_silu, 1, df, d_dn_w, dm, d_contrib, ewgts[k], 0.0f);

        // Add to output
        { int block = 256; int grid = (dm + block - 1) / block;
          add_kernel<<<grid, block, 0, stream>>>(d_contrib, d_out, dm); }
    }
}
// ================================================================
__global__ void softmax_kernel(float *x, int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;
    float *r = x + row * D;

    // Find max
    float mx = -1e30f;
    for (int j = 0; j < D; j++) if (r[j] > mx) mx = r[j];

    // Exp and sum
    float sum = 0.0f;
    for (int j = 0; j < D; j++) { float e = expf(r[j] - mx); r[j] = e; sum += e; }

    // Normalize
    float inv = (sum > 0.0f) ? 1.0f / sum : 1.0f;
    for (int j = 0; j < D; j++) r[j] *= inv;
}

void wubu_cuda_softmax(float *x, int N, int D, cudaStream_t stream) {
    softmax_kernel<<<N, 1, 0, stream>>>(x, N, D);
}

// ================================================================
// Online softmax tile kernel: process 1 row of scores against tile
// scores: [T_tile] raw dot products (in/out: exp(s*scale - M))
// M: [1] running max (in/out)
// L: [1] running sum (in/out)
// O: [hd] accumulated output (rescaled in-place)
// ================================================================
__global__ void online_softmax_row_kernel(float *scores, int T_tile, float scale,
                                          float *M, float *L, float *O, int hd) {
    // Scale scores and find max
    float mx = -1e30f;
    for (int j = 0; j < T_tile; j++) {
        float s = scores[j] * scale;
        scores[j] = s;
        if (s > mx) mx = s;
    }
    float m_old = *M;
    float m_comb = fmaxf(m_old, mx);

    // Rescale old: O *= exp(m_old - m_comb), L *= exp(m_old - m_comb)
    float rescale = expf(m_old - m_comb);
    for (int j = 0; j < hd; j++) O[j] *= rescale;
    float L_old = *L * rescale;

    // Exp and sum new tile
    float sum = 0.0f;
    for (int j = 0; j < T_tile; j++) {
        float e = expf(scores[j] - m_comb);
        scores[j] = e;
        sum += e;
    }

    *M = m_comb;
    *L = L_old + sum;
}

// Normalize: O[row*hd + j] /= L[row] for all rows
__global__ void normalize_attn_kernel(float *O, const float *L, int n_rows, int hd) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    float inv = 1.0f / (L[row] + 1e-30f);
    float *o = O + (size_t)row * hd;
    for (int j = 0; j < hd; j++) o[j] *= inv;
}

// ================================================================
// Chunked attention — tiled version supporting T_cache up to 256K.
//
// Q_chunk [C, N_Q_HEADS * HEAD_DIM] — already RMSNorm'd + RoPE'd
// K_cache [T_cache, N_KV_HEADS * HEAD_DIM] — already RMSNorm'd + RoPE'd
// V_cache [T_cache, N_KV_HEADS * HEAD_DIM]
// out     [C, N_Q_HEADS * HEAD_DIM] — pre-gate attention output (overwrites Q buffer)
//
// Uses cuBLAS SGEMM per Q-head against cached K,V.
// For T_cache > T_TILE (4096), tiles internally with online softmax.
// Score scratch required: C * T_TILE * sizeof(float) (much smaller than old: C*n_q*T_cache)
// ================================================================
#define ATTEN_TILE 4096

size_t wubu_cuda_chunked_attn_query_scratch(int C, int T_max) {
    (void)T_max;
    // Much smaller: only need per-tile scores [C, ATTEN_TILE]
    size_t s = (size_t)C * ATTEN_TILE * sizeof(float);
    // Plus M, L arrays [C * n_q]
    s += (size_t)C * GQA_Q_HEADS * 2 * sizeof(float);
    return s;
}

void wubu_cuda_chunked_attn(cublasHandle_t handle, cudaStream_t stream,
    int C, int T_cache,
    const float *d_Q_chunk,   // [C, N_Q_HEADS * HEAD_DIM] RMSNorm'd + RoPE'd
    const float *d_K_cache,   // [T_cache, N_KV_HEADS * HEAD_DIM] RMSNorm'd + RoPE'd
    const float *d_V_cache,   // [T_cache, N_KV_HEADS * HEAD_DIM]
    const float *d_gate_full, // [C, N_Q_HEADS * HEAD_DIM] raw gate (pre-sigmoid)
    const float *d_output_w,  // [N_Q_HEADS * HEAD_DIM, D_MODEL]
    float *d_out,             // [C, D_MODEL] — final output after gate + output proj
    float *d_score_scratch)   // scratch for per-tile scores + M/L
{
    const int n_q = GQA_Q_HEADS;
    const int n_kv = GQA_KV_HEADS;
    const int gs = n_q / n_kv;
    const int hd = GQA_HEAD_DIM;
    const int q_dim = n_q * hd;
    const int kv_dim = n_kv * hd;
    const float s_scale = 1.0f / sqrtf((float)hd);
    const float alpha = 1.0f, beta = 0.0f;

    float *d_attn_out = (float*)d_Q_chunk; // reuse Q buffer for attention output

    // Partition scratch: [scores_tile(C, ATT_TILE), M(C*n_q), L(C*n_q)]
    float *d_scores_tile = d_score_scratch;
    float *d_M = d_scores_tile + (size_t)C * ATTEN_TILE;
    float *d_L = d_M + (size_t)C * n_q;

    if (T_cache <= ATTEN_TILE) {
        // === DIRECT: single pass (same as before, no tiling) ===
        // Step 1: scores via cuBLAS for all heads at once
        for (int h_kv = 0; h_kv < n_kv; h_kv++) {
            for (int h_off = 0; h_off < gs; h_off++) {
                int h_q = h_kv * gs + h_off;
                float *d_sh = d_scores_tile + (size_t)h_q * T_cache;
                cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    T_cache, C, hd, &s_scale,
                    d_K_cache + h_kv * hd, kv_dim,
                    d_Q_chunk + h_q * hd, q_dim, &beta,
                    d_sh, (size_t)n_q * T_cache);
            }
        }
        // Step 2: softmax
        for (int i = 0; i < C * n_q; i++)
            softmax_kernel<<<1, 1, 0, stream>>>(d_scores_tile + (size_t)i * T_cache, 1, T_cache);
        // Step 3: weighted sum with V
        for (int h_kv = 0; h_kv < n_kv; h_kv++) {
            for (int h_off = 0; h_off < gs; h_off++) {
                int h_q = h_kv * gs + h_off;
                float *d_sh = d_scores_tile + (size_t)h_q * T_cache;
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    hd, C, T_cache, &alpha,
                    d_V_cache + h_kv * hd, kv_dim,
                    d_sh, (size_t)n_q * T_cache, &beta,
                    d_attn_out + h_q * hd, q_dim);
            }
        }
    } else {
        // === TILED: process T_cache in ATTEN_TILE chunks ===
        // Initialize M = -inf, L = 0
        int n_ml = C * n_q;
        {   int block = 256;
            int grid = (n_ml + block - 1) / block;
            init_neg_inf<<<grid, block, 0, stream>>>(d_M, n_ml);
            init_zero_kernel<<<grid, block, 0, stream>>>(d_L, n_ml);
            // Zero out attn_out
            init_zero_kernel<<<(C * q_dim + block - 1) / block, block, 0, stream>>>(d_attn_out, C * q_dim);
        }

        int n_tiles = (T_cache + ATTEN_TILE - 1) / ATTEN_TILE;
        for (int ti = 0; ti < n_tiles; ti++) {
            int t_start = ti * ATTEN_TILE;
            int t_tile = T_cache - t_start;
            if (t_tile > ATTEN_TILE) t_tile = ATTEN_TILE;

            for (int h_kv = 0; h_kv < n_kv; h_kv++) {
                for (int h_off = 0; h_off < gs; h_off++) {
                    int h_q = h_kv * gs + h_off;

                    // SGEMM: scores [C, t_tile] = Q_h [C, hd] @ K_tile^T [hd, t_tile]
                    // scores_col[t_tile, C] with ld = (size_t)n_q * t_tile? 
                    // Actually for single head with tiling, just use ld = t_tile
                    // The scores are stored in d_scores_tile with layout [C, t_tile] row-major
                    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        t_tile, C, hd, &alpha, // alpha=1, apply scale in kernel
                        d_K_cache + (size_t)t_start * kv_dim + h_kv * hd, kv_dim,
                        d_Q_chunk + h_q * hd, q_dim, &beta,
                        d_scores_tile, t_tile);
                    cudaStreamSynchronize(stream);

                    // Online softmax for each row of this tile
                    for (int c = 0; c < C; c++) {
                        int row_idx = c * n_q + h_q;
                        online_softmax_row_kernel<<<1, 1, 0, stream>>>(
                            d_scores_tile + (size_t)c * t_tile, t_tile, s_scale,
                            d_M + row_idx, d_L + row_idx,
                            d_attn_out + (size_t)row_idx * hd, hd);
                    }

                    // SGEMM: attn_contrib [C, hd] = scores [C, t_tile] @ V_tile [t_tile, hd]
                    // Need contiguous layout for cuBLAS: scores [C, t_tile] row-major
                    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        hd, C, t_tile, &alpha,
                        d_V_cache + (size_t)t_start * kv_dim + h_kv * hd, kv_dim,
                        d_scores_tile, t_tile, &alpha, // ADD to (not replace) attn_out
                        d_attn_out + h_q * hd, q_dim);
                }
            }
        }

        // Final normalize: attn_out[row, :] /= L[row]
        int block = 256;
        int grid = (n_ml + block - 1) / block;
        normalize_attn_kernel<<<grid, block, 0, stream>>>(d_attn_out, d_L, n_ml, hd);
        cudaStreamSynchronize(stream);
    }

    // --- Gate multiply: attn_out *= sigmoid(gate) ---
    wubu_cuda_gqa_gate(d_attn_out, d_gate_full, C, q_dim, stream);

    // --- Output projection → d_out [C, D_MODEL] ---
    wubu_cuda_matmul(handle, d_attn_out, C, q_dim, d_output_w, D_MODEL, d_out, 1.0f, 0.0f);

    cudaStreamSynchronize(stream);
}