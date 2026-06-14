/**
 * Optimized Q4_0 Flash Attention Decode - Host API
 */

#ifndef FLASH_ATTN_Q4_0_OPT_CUH
#define FLASH_ATTN_Q4_0_OPT_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

void launch_flash_attn_q4_0_paged_decode_opt(
    const half *Q, const int *block_table, const uint8_t *K_pool, const uint8_t *V_pool, half *O,
    float softmax_scale, bool causal_mask, int window_size,
    int B, int Tq, int Tk, int q_tile_size,
    cudaStream_t stream
);

void launch_flash_attn_q4_0_decode_opt_contiguous(
    const half *Q, const uint8_t *K_blocks, const uint8_t *V_blocks, half *O,
    float softmax_scale, bool causal_mask, int window_size,
    int B, int Tk, int T_capacity,
    cudaStream_t stream
);

#endif // FLASH_ATTN_Q4_0_OPT_CUH