/*
 * wubu_nf4.h — NormalFloat 4-bit quantization (QLoRA paper arXiv:2305.14314)
 *
 * NF4 uses 16 hardcoded quantization levels from the normal distribution
 * quantiles, providing optimal rate-distortion for Gaussian-distributed data.
 * Levels are denser near zero (where most weights cluster) and sparser at
 * the tails, maximizing information entropy per 4-bit code.
 *
 * Block layout: per-block scale (float) + packed 4-bit indices (2 per byte).
 * Block size = 32 elements → 1 float scale + 16 bytes indices = 20 bytes
 * Compression: 32*4=128 bytes F32 → 20 bytes = 6.4x compression
 *
 * C11 only. Opaque structs. No god headers.
 */
#ifndef WUBU_NF4_H
#define WUBU_NF4_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* NF4 block: 32 elements → 1 scale float + 16 packed bytes (2× 4-bit) */
#define WUBU_NF4_BLOCK_SIZE 32
#define WUBU_NF4_PACKED_BYTES (WUBU_NF4_BLOCK_SIZE / 2)

typedef struct {
    float scale;                         /* absmax / 1.0 */
    uint8_t packed[WUBU_NF4_PACKED_BYTES]; /* 2 nibbles per byte */
} __attribute__((packed)) wubu_nf4_block;

/* The 16 hardcoded NF4 quantization levels (from QLoRA paper Appendix E) */
static const float WUBU_NF4_LEVELS[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f
};

/* Midpoints between NF4 levels for binning */
static const float WUBU_NF4_MIDPOINTS[15] = {
    -0.8480964004993439f,
    -0.6106329262256317f,
    -0.4599952697753906f,
    -0.3396794348955154f,
    -0.2346074059605598f,
    -0.1379117332403894f,
    -0.045525018125772475f,
    0.03979014977812767f,
    0.1202552504837513f,
    0.2040212516784749f,
    0.2920137718319898f,
    0.3893125355243683f,
    0.5016634166241682f,
    0.6427869200706482f,
    0.8614784181118012f
};

/* Quantize one block of 32 floats into NF4 format.
 * input:  [32] F32 values
 * output: wubu_nf4_block (scale + packed nibbles) */
void wubu_nf4_quantize_block(const float *input, wubu_nf4_block *out);

/* Dequantize one NF4 block back to 32 floats.
 * block:  NF4 block
 * output: [32] F32 values */
void wubu_nf4_dequantize_block(const wubu_nf4_block *block, float *output);

/* Quantize a full vector of n elements (n must be multiple of 32).
 * input:  [n] F32
 * output: [n/32] NF4 blocks */
void wubu_nf4_quantize(const float *input, wubu_nf4_block *output, int n);

/* Dequantize a full vector of n elements.
 * blocks: [n/32] NF4 blocks
 * output: [n] F32 */
void wubu_nf4_dequantize(const wubu_nf4_block *blocks, float *output, int n);

/* Fused NF4 dequant + dot product: dot(q, dequantize(blocks))
 * q:      [n] F32 query
 * blocks: [n/32] NF4 blocks
 * Returns: scalar dot product */
float wubu_nf4_dequant_dot(const float *q, const wubu_nf4_block *blocks, int n);

/* Storage size in bytes for n elements */
static inline int wubu_nf4_storage_bytes(int n) {
    return ((n + WUBU_NF4_BLOCK_SIZE - 1) / WUBU_NF4_BLOCK_SIZE) * (int)sizeof(wubu_nf4_block);
}

/* Compression ratio vs F32 */
static inline float wubu_nf4_compression_ratio(void) {
    return (float)(WUBU_NF4_BLOCK_SIZE * 4) / (float)sizeof(wubu_nf4_block);
}

#ifdef __cplusplus
}
#endif
#endif /* WUBU_NF4_H */
