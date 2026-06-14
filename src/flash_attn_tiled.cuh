/**
 * Flash Attention Tiled Kernel for bytropix
 *
 * Based on llama.cpp ggml-cuda/fattn-tile.cuh but adapted for standalone C11+C11 compatibility.
 * Supports:
 * - head_dim = 256 (Qwen3.6, Gemma 4 sliding window)
 * - head_dim = 512 (Gemma 4 full attention)
 * - GQA support (n_q_heads = 16, n_kv_heads = 2 for Qwen; 16/8 for Gemma SW; 16/1 for Gemma full)
 * - Sliding window attention
 * - Logit softcapping
 * - Online softmax with warp-level reduction
 *
 * Uses warp-level online softmax and shared memory tiling.
 * Optimized for Volta+ tensor cores (via WMMA/MMA in future).
 */

#ifndef FLASH_ATTN_TILED_CUH
#define FLASH_ATTN_TILED_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ================================================================
// Configuration constants
// ================================================================

// Tile sizes (matching llama.cpp for head_dim=256)
#define FLASH_ATTN_TILE_KV     128   // ATTEN_TILE = 16384 / 128 = 128 tiles max
#define FLASH_ATTN_TILE_Q      128   // Q tiles
#define WARP_SIZE              32
#define MMA_M                  16
#define MMA_N                  8
#define MMA_K                  16

// GQA constants
#define MAX_N_Q_HEADS          16
#define MAX_N_KV_HEADS         8
#define MAX_HEAD_DIM           512
#define MAX_N_Q_HEADS_PER_KV   16  // max Q heads per KV head

// Derived
#define MMA_ROWS_PER_WARP      (MMA_M / WARP_SIZE)  // 16/32 = 0.5 -> 2 warps per MMA_M
#define MMA_COLS_PER_WARP      (MMA_N / WARP_SIZE)  // 8/32 = 0.25

// ================================================================
// Interface
// ================================================================

// Sanitize NaN in FP16 tensor
void flash_attn_sanitize_f16(half *d_ptr, int n, cudaStream_t stream);

// Launch flash attention for head_dim=256 (Qwen, Gemma SW)
void launch_flash_attn_256(
    const half *Q, const half *K, const half *V, half *O,
    float softmax_scale, bool causal_mask, int window_size, float logit_softcap,
    int B, int Hq, int Hkv, int Tq, int Tk,
    cudaStream_t stream
);

// Launch flash attention for head_dim=512 (Gemma full attention)
void launch_flash_attn_512(
    const half *Q, const half *K, const half *V, half *O,
    float softmax_scale, bool causal_mask, int window_size, float logit_softcap,
    int B, int Hq, int Hkv, int Tq, int Tk,
    cudaStream_t stream
);

#endif // FLASH_ATTN_TILED_CUH