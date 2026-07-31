/*
 * wubu_nf4.c — NormalFloat 4-bit quantization implementation
 *
 * NF4 uses 16 quantization levels from normal distribution quantiles.
 * Quantization: normalize to [-1,1] via block absmax, then nearest-level
 * lookup via binary search over midpoint bins.
 * Dequantization: scale * NF4_LEVELS[nibble]
 *
 * C11 only. Self-contained. No external dependencies.
 */
#include "wubu_nf4.h"
#include <string.h>
#include <math.h>
#include <stdint.h>

/* Linear search for NF4 level index given normalized value in [-1,1].
 * 15 comparisons — branch predictor handles this well, and avoids
 * the binary search bug where midpoints were miscomputed. */
static inline int nf4_find_level(float x) {
    /* Clamp */
    if (x <= WUBU_NF4_LEVELS[0]) return 0;
    if (x >= WUBU_NF4_LEVELS[15]) return 15;
    for (int i = 0; i < 15; i++) {
        float mid = (WUBU_NF4_LEVELS[i] + WUBU_NF4_LEVELS[i+1]) * 0.5f;
        if (x <= mid) return i;
    }
    return 15;
}

void wubu_nf4_quantize_block(const float *input, wubu_nf4_block *out)
{
    /* Find block absmax */
    float max_abs = 0.0f;
    for (int i = 0; i < WUBU_NF4_BLOCK_SIZE; i++) {
        float a = fabsf(input[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-10f) max_abs = 1e-10f;
    out->scale = max_abs;

    /* Normalize and quantize to NF4 levels */
    float inv_scale = 1.0f / max_abs;
    for (int i = 0; i < WUBU_NF4_BLOCK_SIZE; i += 2) {
        float n0 = input[i] * inv_scale;
        float n1 = input[i+1] * inv_scale;
        int idx0 = nf4_find_level(n0);
        int idx1 = nf4_find_level(n1);
        out->packed[i/2] = (uint8_t)((idx0 << 4) | idx1);
    }
}

void wubu_nf4_dequantize_block(const wubu_nf4_block *block, float *output)
{
    float s = block->scale;
    for (int i = 0; i < WUBU_NF4_PACKED_BYTES; i++) {
        uint8_t byte = block->packed[i];
        int idx0 = (byte >> 4) & 0xF;
        int idx1 = byte & 0xF;
        output[i*2]   = s * WUBU_NF4_LEVELS[idx0];
        output[i*2+1] = s * WUBU_NF4_LEVELS[idx1];
    }
}

void wubu_nf4_quantize(const float *input, wubu_nf4_block *output, int n)
{
    int n_blocks = (n + WUBU_NF4_BLOCK_SIZE - 1) / WUBU_NF4_BLOCK_SIZE;
    for (int b = 0; b < n_blocks; b++) {
        int offset = b * WUBU_NF4_BLOCK_SIZE;
        int remaining = n - offset;
        if (remaining >= WUBU_NF4_BLOCK_SIZE) {
            wubu_nf4_quantize_block(input + offset, &output[b]);
        } else {
            /* Pad last block with zeros */
            float pad[WUBU_NF4_BLOCK_SIZE];
            memset(pad, 0, sizeof(pad));
            memcpy(pad, input + offset, (size_t)remaining * sizeof(float));
            wubu_nf4_quantize_block(pad, &output[b]);
        }
    }
}

void wubu_nf4_dequantize(const wubu_nf4_block *blocks, float *output, int n)
{
    int n_blocks = (n + WUBU_NF4_BLOCK_SIZE - 1) / WUBU_NF4_BLOCK_SIZE;
    for (int b = 0; b < n_blocks; b++) {
        float tmp[WUBU_NF4_BLOCK_SIZE];
        wubu_nf4_dequantize_block(&blocks[b], tmp);
        int offset = b * WUBU_NF4_BLOCK_SIZE;
        int remaining = n - offset;
        int copy_n = remaining < WUBU_NF4_BLOCK_SIZE ? remaining : WUBU_NF4_BLOCK_SIZE;
        memcpy(output + offset, tmp, (size_t)copy_n * sizeof(float));
    }
}

float wubu_nf4_dequant_dot(const float *q, const wubu_nf4_block *blocks, int n)
{
    float dot = 0.0f;
    int n_blocks = (n + WUBU_NF4_BLOCK_SIZE - 1) / WUBU_NF4_BLOCK_SIZE;
    for (int b = 0; b < n_blocks; b++) {
        float s = blocks[b].scale;
        int offset = b * WUBU_NF4_BLOCK_SIZE;
        for (int i = 0; i < WUBU_NF4_PACKED_BYTES; i++) {
            uint8_t byte = blocks[b].packed[i];
            int idx0 = (byte >> 4) & 0xF;
            int idx1 = byte & 0xF;
            int j0 = offset + i*2;
            int j1 = offset + i*2 + 1;
            if (j0 < n) dot += q[j0] * s * WUBU_NF4_LEVELS[idx0];
            if (j1 < n) dot += q[j1] * s * WUBU_NF4_LEVELS[idx1];
        }
    }
    return dot;
}
