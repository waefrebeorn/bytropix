/* wubu_enc_h3.c — MiniMax H3 text encoder NVFP4 requant + ConvRot un-rotation
 *
 * The MiniMax H3 encoder stores ConvRot weights in rotated form
 * (Hadamard applied to the weight prefix). To re-quantize to
 * NVFP4, the weights must be un-rotated FIRST — otherwise the
 * output is unrelated to the prompt.
 *
 * Since Hadamard is self-inverse (H·H = I), un-rotation = rotation.
 * We reuse wubu_hadamard() from wubu_rotate.c.
 *
 * C11, opaque struct, minimal includes. No third-party deps.
 */
#define _POSIX_C_SOURCE 200809L
#include "wubu_enc_h3.h"
#include "wubu_rotate.h"
#include "wubu_nvfp4.h"
#include "wubu_fp8.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct wubu_enc_h3 {
    int rows;
    int cols;
    int prefix; /* P = pow2_floor(cols) — the ConvRot prefix */
};

wubu_enc_h3_t *wubu_enc_h3_create(int rows, int cols) {
    wubu_enc_h3_t *enc = (wubu_enc_h3_t *)calloc(1, sizeof(*enc));
    if (!enc) return NULL;
    enc->rows = rows;
    enc->cols = cols;
    enc->prefix = wubu_pow2_floor(cols);
    return enc;
}

/* Un-rotate the ConvRot prefix of W[rows x cols].
 * Applies Hadamard to the first P columns of each row.
 * Hadamard is self-inverse, so this undoes the rotation. */
int wubu_enc_h3_unrotate(float *W, int rows, int cols) {
    if (!W || rows <= 0 || cols <= 0) return -1;
    int P = wubu_pow2_floor(cols);
    if (P <= 1) return 0; /* nothing to rotate */
    float *tmp = (float *)malloc((size_t)P * sizeof(float));
    if (!tmp) return -1;
    for (int r = 0; r < rows; r++) {
        float *wr = W + (size_t)r * cols;
        memcpy(tmp, wr, (size_t)P * sizeof(float));
        wubu_hadamard(tmp, P);
        memcpy(wr, tmp, (size_t)P * sizeof(float));
    }
    free(tmp);
    return 0;
}

/* Packed NVFP4 size for a [rows x cols] matrix.
 * NVFP4: 2 elements per byte (4 bits each), plus 1 FP8 E4M3 scale
 * byte per block of 32 elements. */
size_t wubu_enc_h3_packed_size(int rows, int cols) {
    int total_elements = rows * cols;
    int n_blocks = (total_elements + 31) / 32;
    /* Each block: 16 bytes of packed nibbles + 1 byte FP8 scale */
    return (size_t)n_blocks * 17;
}

int wubu_enc_h3_requant_nvfp4(const float *W, int rows, int cols,
                                          uint8_t *out_packed, size_t packed_size) {
    if (!W || !out_packed || rows <= 0 || cols <= 0) return -1;
    size_t expected = wubu_enc_h3_packed_size(rows, cols);
    if (packed_size < expected) return -1;

    int total = rows * cols;
    int n_blocks = (total + 31) / 32;

    /* Use wubu_nvfp4_block_quantize for the whole matrix.
     * It writes raw nibbles to packed and FP8 scales to scale_out.
     * We then interleave them: [scale_byte, 16 nibbles] per block. */
    uint8_t *raw_nibbles = (uint8_t *)malloc((size_t)n_blocks * 16);
    uint8_t *scales = (uint8_t *)malloc((size_t)n_blocks);
    if (!raw_nibbles || !scales) { free(raw_nibbles); free(scales); return -1; }

    float *tmp = (float *)malloc((size_t)total * sizeof(float));
    if (!tmp) { free(raw_nibbles); free(scales); return -1; }
    memcpy(tmp, W, (size_t)total * sizeof(float));

    int nb = wubu_nvfp4_block_quantize(tmp, raw_nibbles, scales, total, 32);
    (void)nb;

    /* Interleave: for each block, write 1 FP8 scale byte + 16 nibbles */
    uint8_t *out = out_packed;
    for (int bk = 0; bk < n_blocks; bk++) {
        *out++ = scales[bk];
        int nbytes = (bk == n_blocks - 1)
            ? ((total - bk * 32 + 1) / 2)
            : 16;
        memcpy(out, raw_nibbles + bk * 16, (size_t)nbytes);
        out += nbytes;
    }

    free(raw_nibbles);
    free(scales);
    free(tmp);
    return 0;
}

/* Dequant NVFP4 packed data back to float, then re-rotate (Hadamard again).
 * The dequantized+rotated result is written to W_out[rows x cols].
 *
 * Packed format: [scale_byte, 16 nibbles] per 32-element block.
 * The quant writes nibble at byte = start/2 + i/2 (row-major within block).
 * The dequant must read from the same offset. */
int wubu_enc_h3_dequant_rotate(const uint8_t *packed, int rows, int cols,
                                           float *W_out, size_t packed_size) {
    if (!packed || !W_out || rows <= 0 || cols <= 0) return -1;
    int P = wubu_pow2_floor(cols);
    int total = rows * cols;
    int n_blocks = (total + 31) / 32;

    /* Extract scales and raw nibbles */
    uint8_t *scales = (uint8_t *)malloc((size_t)n_blocks);
    uint8_t *raw_nibbles = (uint8_t *)malloc((size_t)n_blocks * 16);
    if (!scales || !raw_nibbles) { free(scales); free(raw_nibbles); return -1; }

    const uint8_t *src = packed;
    for (int bk = 0; bk < n_blocks; bk++) {
        scales[bk] = *src++;
        int nbytes = (bk == n_blocks - 1)
            ? ((total - bk * 32 + 1) / 2)
            : 16;
        memcpy(raw_nibbles + bk * 16, src, (size_t)nbytes);
        src += nbytes;
    }

    /* Dequant using NVFP4 E2M1 format:
     * Each nibble is a floating-point value decoded by wubu_nvfp4_to_f32,
     * then multiplied by the FP8 E4M3 block scale. */
    float *tmp = (float *)malloc((size_t)total * sizeof(float));
    if (!tmp) { free(scales); free(raw_nibbles); return -1; }

    for (int bk = 0; bk < n_blocks; bk++) {
        int start = bk * 32;
        int cnt = (bk == n_blocks - 1) ? (total - start) : 32;
        float scale = wubu_fp8_e4m3_to_f32(scales[bk]);
        if (scale < 1e-9f) scale = 1e-9f;
        for (int i = 0; i < cnt; i++) {
            /* Same byte layout as wubu_nvfp4_block_quantize:
             * byte = start/2 + i/2 (absolute within raw_nibbles),
             * even i → low nibble, odd i → high nibble. */
            int byte_idx = start / 2 + i / 2;
            int nibble_idx = i % 2;
            uint8_t nib = (nibble_idx == 0)
                ? (raw_nibbles[byte_idx] & 0xF)
                : ((raw_nibbles[byte_idx] >> 4) & 0xF);
            float q = wubu_nvfp4_to_f32(nib);
            tmp[start + i] = q * scale;
        }
    }

    /* Copy to output and re-rotate the prefix */
    memcpy(W_out, tmp, (size_t)total * sizeof(float));
    if (P > 1) {
        for (int r = 0; r < rows; r++) {
            wubu_hadamard(W_out + (size_t)r * cols, P);
        }
    }

    free(scales);
    free(raw_nibbles);
    free(tmp);
    return 0;
}

void wubu_enc_h3_free(wubu_enc_h3_t *enc) {
    if (!enc) return;
    free(enc);
}
