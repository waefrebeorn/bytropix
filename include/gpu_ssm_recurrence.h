#ifndef WUBU_GPU_SSM_RECURRENCE_H
#define WUBU_GPU_SSM_RECURRENCE_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Repeat K heads 16→32 for SSM recurrence input.
// q_in/k_in: [K_HEADS=16][D_STATE=128] on GPU
// v_in: [V_HEADS=32][D_STATE=128] on GPU  
// beta_in/gate_in: [DT_RANK=32] on GPU
// q_out/k_out/v_out: [V_HEADS=32][D_STATE=128] on GPU (output)
// beta_out/gate_out: [V_HEADS=32] on GPU (output)
void wubu_gpu_repeat_kheads(
    const float *q_in, const float *k_in, const float *v_in,
    const float *beta_in, const float *gate_in,
    float *q_out, float *k_out, float *v_out,
    float *beta_out, float *gate_out,
    cudaStream_t stream);

// Run SSM recurrence for one token on GPU.
// All pointers are device pointers.
// ssm_state: [V_HEADS][D_STATE][D_STATE] — persistent, updated in-place
// q/k/v: [V_HEADS][D_STATE]
// beta/gate: [V_HEADS]
// delta_out: [V_HEADS][D_STATE]
void wubu_gpu_ssm_recurrence(
    float *ssm_state,
    const float *q, const float *k, const float *v,
    const float *beta, const float *gate,
    float *delta_out,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // WUBU_GPU_SSM_RECURRENCE_H
