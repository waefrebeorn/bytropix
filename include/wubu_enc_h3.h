/* wubu_enc_h3.h — MiniMax H3 text encoder NVFP4 requant + ConvRot un-rotation
 *
 * The MiniMax H3 encoder stores ConvRot weights in rotated form
 * (Hadamard applied to the weight prefix). To re-quantize to
 * NVFP4, the weights must be un-rotated FIRST — otherwise the
 * output is unrelated to the prompt.
 *
 * Pipeline:
 *   1. wubu_enc_h3_unrotate() — undo the ConvRot rotation (Hadamard)
 *   2. wubu_enc_h3_requant_nvfp4() — re-quantize un-rotated weights to NVFP4
 *   3. At inference: dequant NVFP4 → rotate back (Hadamard again)
 *
 * Since Hadamard is self-inverse (H·H = I), un-rotation = rotation.
 * We reuse wubu_hadamard() from wubu_rotate.c.
 *
 * C11, opaque struct, minimal includes. No third-party deps.
 *
 * Reference: DiffSynth-Studio/MiniMax-H3-NF4 (modelscope)
 *   26.4 GB → 15.7 GB, runs on single 16 GB card (peak 9.9 GB VRAM)
 *   Drop-in replacement for Comfy-Org NVFP4 encoder
 */

#ifndef WUBU_ENC_H3_H
#define WUBU_ENC_H3_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque H3 encoder handle */
typedef struct wubu_enc_h3 wubu_enc_h3_t;

/* Create an H3 encoder context for a weight matrix of [rows x cols].
 * The ConvRot prefix is the largest power-of-2 <= cols.
 * Returns NULL on allocation failure. */
wubu_enc_h3_t *wubu_enc_h3_create(int rows, int cols);

/* Un-rotate the ConvRot prefix of weight matrix W[rows x cols].
 * Applies Hadamard to the first P = pow2_floor(cols) columns of each row,
 * undoing the ConvRot rotation so the weights can be re-quantized.
 * This is the SAME operation as wubu_rotate_fuse_right (Hadamard is
 * self-inverse: H·H = I).
 * Returns 0 on success, -1 on error. */
int wubu_enc_h3_unrotate(float *W, int rows, int cols);

/* Re-quantize un-rotated weights to NVFP4 in-place.
 * The weight matrix W[rows x cols] (float) is quantized to NVFP4
 * and stored in out_packed. Returns 0 on success, -1 on error.
 * Caller must allocate out_packed of size wubu_enc_h3_packed_size(rows, cols). */
int wubu_enc_h3_requant_nvfp4(const float *W, int rows, int cols,
                                  uint8_t *out_packed, size_t packed_size);

/* Compute the packed NVFP4 size for a [rows x cols] weight matrix. */
size_t wubu_enc_h3_packed_size(int rows, int cols);

/* Dequant NVFP4 back to float, then re-rotate (Hadamard again).
 * This is the inference path: dequant → rotate → use in GEMV.
 * The dequantized+rotated result is written to W_out[rows x cols].
 * Returns 0 on success, -1 on error. */
int wubu_enc_h3_dequant_rotate(const uint8_t *packed, int rows, int cols,
                                   float *W_out, size_t packed_size);

/* Destroy the H3 encoder context. */
void wubu_enc_h3_free(wubu_enc_h3_t *enc);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_ENC_H3_H */
