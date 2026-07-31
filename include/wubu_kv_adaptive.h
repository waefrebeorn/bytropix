/*
 * wubu_kv_adaptive.h -- Ecco entropy-aware adaptive KV compression (doc 001).
 *
 * Per-block bit-width selection based on variance (entropy proxy):
 *   - Low variance blocks → 2-4 bits (save bandwidth)
 *   - High variance blocks → 8 bits (preserve accuracy)
 *
 * Self-contained C11, no third-party deps.
 *
 * Basis: Cheng et al., "Ecco", ISCA 2025.
 */

#ifndef WUBU_KV_ADAPTIVE_H
#define WUBU_KV_ADAPTIVE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Quantize n floats to adaptive-width packed bytes.
 * Stores: width_bits (2/4/8), scale (absmax), packed[n] (uint8 per element).
 * Returns 0 on success, -1 on error. */
int wubu_kvq_adaptive_quant(const float *z, uint8_t *out, int *width_bits,
                               float *out_scale, int n);

/* Dequantize packed bytes back to floats using the stored width and scale.
 * Returns 0 on success, -1 on error. */
int wubu_kvq_adaptive_dequant(const uint8_t *packed, int width_bits,
                                float scale, float *out, int n);

/* Round-trip test helper: quantize → dequantize and return cosine similarity.
 * Returns cosine in [0, 1], or 0 on error. */
float wubu_kvq_adaptive_roundtrip(const float *z, int n);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_KV_ADAPTIVE_H */
