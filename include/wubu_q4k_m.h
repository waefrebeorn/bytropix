#ifndef WUBU_Q4_K_M_H
#define WUBU_Q4_K_M_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Q4_K_M quantization - the community default for 4-bit LLM deployment.
 *
 * CONVERGENT WIN (Kevin-Bacon meta-analysis):
 *   - llama.cpp: Q4_K_M is THE default for ≤16GB VRAM (~4.5 bpw, ~1-3% quality loss)
 *   - K-quant PR #1684: super-blocks (256), double-quant scales, mixed precision
 *   - imatrix calibration: IQ4_XS/IQ3_M best quality/byte with good imatrix
 *   - PromptQuorum 2026: Q4_K_M "widely useful default"
 *
 * Layout (matching GGUF Q4_K):
 *   - Block size: 256 weights (super-block) = 8 sub-blocks of 32
 *   - Per super-block: 2 fp16 scales (d, dmin) + 12 uint8 scales (sc) + 128 uint4 quants (qs)
 *   - Mixed precision: sensitive tensors (attn_v, output, ffn_down) at 6-bit via Q4_K_M "M"
 *
 * Per-sub-block (32 weights):
 *   - quant = round(x / scale) clamped to [-8, 7] (signed 4-bit)
 *   - scale = d * sc[i] / 255 where d = max(abs(x)) / 8
 *
 * The "M" variant uses 6-bit for sensitive layers: attn_v, output_proj, ffn_down
 */

#define Q4K_SUPER_BLOCK 256
#define Q4K_SUB_BLOCK 32
#define Q4K_SUB_PER_SUPER 8

/* Quantization types within Q4_K_M */
typedef enum {
    WUBU_Q4K_NONE = 0,    /* no quantization */
    WUBU_Q4K_BASE = 1,    /* standard Q4_K (all 4-bit) */
    WUBU_Q4K_M = 2        /* mixed: sensitive tensors at 6-bit */
} wubu_q4k_variant_t;

/* Q4_K block structure (matches GGUF layout) */
typedef struct {
    /* d, dmin: fp16 scales (2 bytes each = 4 bytes) */
    uint16_t d;
    uint16_t dmin;
    /* sc: 12 uint8 scales (12 bytes) */
    uint8_t sc[Q4K_SUB_PER_SUPER];
    /* qs: 128 packed uint4 (64 bytes) */
    uint8_t qs[Q4K_SUPER_BLOCK / 2];
} wubu_q4k_block_t;

/* Quantize n floats to Q4_K_M blocks. n must be multiple of 256.
 * For WUBU_Q4K_M variant: if 'is_sensitive' is true, use 6-bit for that block. */
void wubu_q4k_quant(const float *x, wubu_q4k_block_t *blocks, int n, int is_sensitive);

/* Dequantize Q4_K_M blocks back to float */
void wubu_q4k_dequant(const wubu_q4k_block_t *blocks, float *x, int n);

/* Round-trip accuracy check */
float wubu_q4k_cosine(const float *a, const float *b, int n);

/* Bytes per element for capacity planning */
float wubu_q4k_bytes_per_elem(wubu_q4k_variant_t variant);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_Q4_K_M_H */