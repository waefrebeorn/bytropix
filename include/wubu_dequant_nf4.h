/*
 * wubu_dequant_nf4.h — NF4 (Normal Float 4) dequantization for wubuwizard.
 *
 * NF4 is the 4-bit quantization format used by bitsandbytes (nf4 fp4).
 * Unlike MXFP4/NVFP4 which use E2M1 mantissas + microscaling, NF4 maps
 * each 4-bit code directly to a level of the standard normal distribution
 * via the inverse CDF: code i → Φ^{-1}((2i+1)/32) for i in [0, 15].
 *
 * NF4 has no per-block scale factor; the scale is fused into the weight
 * tensor at quantization time (the quantizer divides by the per-tensor
 * absmax before encoding). At dequant time, the scale is stored as a
 * separate FP32/BF16 scalar adjacent to the packed data.
 *
 * For the MiniMax H3 NF4 model on ModelScope, weights come in as
 * safetensors with NF4-quantized int32 arrays + fp32 scales.
 *
 * This header provides an opaque API consistent with wubu_dequant_fp4.h.
 */
#ifndef WUBU_DEQUANT_NF4_H
#define WUBU_DEQUANT_NF4_H

#include <stddef.h>

/*
 * nf4_dec_dequantize_row — dequantize a row of packed NF4 values.
 *
 * The NF4 format packs two 4-bit codes per byte (high nibble first),
 * i.e. 16 values per 8 bytes. Each code maps to a level of Φ^{-1}:
 *
 *   code → Φ^{-1}((2*code+1)/32)
 *
 * This implementation precomputes the 16-entry lookup table and
 * applies: out[i] = nf4_levels[code_i] * scale
 *
 * Parameters:
 *   __restrict src    — packed NF4 data (high nibble first: code(7:4), code(3:0))
 *   __restrict out    — output float array (n elements)
 *   scale            — per-tensor scale factor (float)
 *   n                — number of elements to dequantize
 */
void nf4_dequantize_row(const unsigned char *src, float *out,
                         float scale, long n);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_DEQUANT_NF4_H */
