#include "cuda_kernels.h"
#include "wubu_ssm.h"
#include <stdio.h>

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
    y[i] = expf(x[i]);
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
// RMSNorm kernel
// Input: x [B, T, d], weight [d]
// Output: out [B, T, d]
// Each sequence element: out[i] = x[i] * weight[i] / sqrt(mean(x²) + eps)
// ================================================================
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

    float egate = expf(gate_raw);

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
        float egate = expf(gate_val);
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
    float *d_scratch) {         // [N, q_dim] — temp for attention out
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
    float gg = expf(d_gate[s * DT_RANK + kh]);
    
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

