#ifndef WUBU_GPU_SSM_RECURRENCE_H
#define WUBU_GPU_SSM_RECURRENCE_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

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

#endif
