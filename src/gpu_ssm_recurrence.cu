/**
 * gpu_ssm_recurrence.cu — GPU SSM selective scan recurrence.
 *
 * One block per V-head (32), 128 threads per block.
 * State matrix h[128][128] in global memory (64KB/head, too large for shared mem).
 * Shared memory: q[128], k[128], v[128], hk[128], diff[128] = 2.5KB.
 *
 * Each thread manages one row of the state matrix.
 * All 32 V-heads run in parallel (32 blocks).
 *
 * New: repeat_k_heads kernel eliminates CPU-based head repetition (16→32).
 */
#include "gpu_ssm_recurrence.h"
#include <cuda_runtime.h>
#include <math.h>

#define V_HEADS 32
#define D_STATE 128
#define K_HEADS 16

// Kernel: repeat K heads 16→32 and pack into [V_HEADS][D_STATE] layout
// q_in:  [K_HEADS][D_STATE] on GPU
// k_in:  [K_HEADS][D_STATE] on GPU
// v_in:  [V_HEADS][D_STATE] on GPU
// beta_in:[DT_RANK] on GPU
// gate_in:[DT_RANK] on GPU
// q_out: [V_HEADS][D_STATE] on GPU (output)
// k_out: [V_HEADS][D_STATE] on GPU (output)
// v_out: [V_HEADS][D_STATE] on GPU (output, same as v_in)
// beta_out:[V_HEADS] on GPU (output, broadcast DT_RANK→V_HEADS)
// gate_out:[V_HEADS] on GPU (output, broadcast DT_RANK→V_HEADS)
__global__ void repeat_kheads_kernel(
    const float * __restrict__ q_in,
    const float * __restrict__ k_in,
    const float * __restrict__ v_in,
    const float * __restrict__ beta_in,
    const float * __restrict__ gate_in,
    float * __restrict__ q_out,
    float * __restrict__ k_out,
    float * __restrict__ v_out,
    float * __restrict__ beta_out,
    float * __restrict__ gate_out)
{
    int vh = blockIdx.x;       // 0..31
    int i  = threadIdx.x;      // 0..127
    int kh = vh % K_HEADS;     // which K-head to repeat
    int idx = vh * D_STATE + i;

    q_out[idx] = q_in[kh * D_STATE + i];
    k_out[idx] = k_in[kh * D_STATE + i];
    v_out[idx] = v_in[idx];    // passthrough (V already has 32 heads)

    if (i == 0) {
        beta_out[vh] = beta_in[vh];
        gate_out[vh] = gate_in[vh];
    }
}

// Recurrence for ONE token. State persists in global memory across calls.
__global__ void ssm_recurrence_kernel(
    float * __restrict__ ssm_state,   // [V_HEADS][D_STATE][D_STATE]
    const float * __restrict__ q_all, // [V_HEADS][D_STATE]
    const float * __restrict__ k_all, // [V_HEADS][D_STATE]
    const float * __restrict__ v_all, // [V_HEADS][D_STATE]
    const float * __restrict__ beta,  // [V_HEADS]
    const float * __restrict__ gate,  // [V_HEADS]
    float * __restrict__ delta_out,   // [V_HEADS][D_STATE]
    float q_scale)
{
    int vh = blockIdx.x;
    int i  = threadIdx.x;

    __shared__ float s_q[D_STATE], s_k[D_STATE], s_v[D_STATE];
    __shared__ float s_hk[D_STATE], s_diff[D_STATE];

    float *h = ssm_state + (int64_t)vh * D_STATE * D_STATE;

    s_q[i] = q_all[(int64_t)vh * D_STATE + i];
    s_k[i] = k_all[(int64_t)vh * D_STATE + i];
    s_v[i] = v_all[(int64_t)vh * D_STATE + i];
    __syncthreads();

    float bg = beta[vh];
    float gg = expf(gate[vh]);

    float *row_i = h + (int64_t)i * D_STATE;
    for (int j = 0; j < D_STATE; j++)
        row_i[j] *= gg;
    __syncthreads();

    float hk_i = 0.0f;
    for (int j = 0; j < D_STATE; j++)
        hk_i += row_i[j] * s_k[j];
    s_hk[i] = hk_i;
    __syncthreads();

    s_diff[i] = s_v[i] - s_hk[i];
    __syncthreads();

    float k_i = s_k[i];
    for (int j = 0; j < D_STATE; j++)
        row_i[j] += bg * k_i * s_diff[j];
    __syncthreads();

    float delta_i = 0.0f;
    for (int j = 0; j < D_STATE; j++)
        delta_i += row_i[j] * s_q[j];

    delta_out[(int64_t)vh * D_STATE + i] = delta_i * q_scale;
}

// Host wrapper for repeat_kheads_kernel
void wubu_gpu_repeat_kheads(
    const float *q_in, const float *k_in, const float *v_in,
    const float *beta_in, const float *gate_in,
    float *q_out, float *k_out, float *v_out,
    float *beta_out, float *gate_out,
    cudaStream_t stream)
{
    repeat_kheads_kernel<<<V_HEADS, D_STATE, 0, stream>>>(
        q_in, k_in, v_in, beta_in, gate_in,
        q_out, k_out, v_out, beta_out, gate_out);
}

// Host wrapper
void wubu_gpu_ssm_recurrence(
    float *ssm_state,
    const float *q, const float *k, const float *v,
    const float *beta, const float *gate,
    float *delta_out,
    cudaStream_t stream)
{
    float q_scale = 1.0f / sqrtf((float)D_STATE);
    ssm_recurrence_kernel<<<V_HEADS, D_STATE, 0, stream>>>(
        ssm_state, q, k, v, beta, gate, delta_out, q_scale);
}
