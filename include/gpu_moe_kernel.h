#ifndef WUBU_GPU_MOE_KERNEL_H
#define WUBU_GPU_MOE_KERNEL_H

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize GPU MoE: copies IQ2_XXS/IQ3_XXS lookup tables to GPU constant memory.
// Must be called once after cudaSetDevice.
void wubu_gpu_moe_init(void);

// Run one token through 8 routed experts on GPU.
// x: [D_MODEL] input (host pointer, uploaded internally)
// gate_q/up_q/down_q: [8] pointers to per-expert quantized weight data (host -> uploaded internally)
// gate_bytes/up_bytes/down_bytes: bytes per expert for each weight matrix
// gate_type/up_type/down_type: GGML quant type (IQ2_XXS or IQ3_XXS)
// weights: [8] routing weights (host)
// output: [8][D_MODEL] output (host, downloaded internally)
// d_gate_buf/d_up_buf/d_down_buf/d_out_buf/d_weights_buf: pre-allocated GPU buffers
void wubu_gpu_moe_forward_experts(
    const float *x,
    const uint8_t *const *gate_q, const int64_t gate_bytes,
    const uint8_t *const *up_q, const int64_t up_bytes,
    const uint8_t *const *down_q, const int64_t down_bytes,
    int gate_type, int up_type, int down_type,
    const float *weights,
    float *output,
    cudaStream_t stream,
    uint8_t *d_gate_buf, uint8_t *d_up_buf, uint8_t *d_down_buf,
    float *d_out_buf, float *d_weights_buf);

#ifdef __cplusplus
}
#endif

#endif
