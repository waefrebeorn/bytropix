/**
 * Q4_0 Flash Attention for Prefill
 * Fused kernel replacing per-tile cublasGEMM loop
 */

#ifndef FLASH_ATTN_Q4_0_CUH
#define FLASH_ATTN_Q4_0_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

void launch_flash_attn_q4_0_prefill(
    const half *Q, const uint8_t *K_blocks, const uint8_t *V_blocks, half *O,
    float softmax_scale, bool causal_mask, int window_size,
    int B, int Tq, int Tk, int q_tile_size,
    cudaStream_t stream
);

void launch_flash_attn_q4_0_decode(
    const half *Q, const uint8_t *K_blocks, const uint8_t *V_blocks, half *O,
    float softmax_scale, bool causal_mask, int window_size,
    int B, int Tq, int Tk, int q_tile_size,
    cudaStream_t stream
);

#endif // FLASH_ATTN_Q4_0_CUH