#ifndef WUBU_DEQUANT_FP4_H
#define WUBU_DEQUANT_FP4_H

#include <stddef.h>

/*
 * wubu_dequant_fp4 — C11 row-level dequantization for OCP MXFP4 and
 * NVFP4 microscaling formats, wired into the GGUF reader pipeline.
 *
 * MXFP4 (type 39): 32-element blocks
 *   1 byte E8M0 shared scale  (bias 127: scale = 2^(E-127))
 *   16 bytes packed 4-bit E2M1  values (2 per byte)
 *   = 17 bytes per 32 elements
 *
 * NVFP4 (type 40): 64-element blocks
 *   4 bytes UE4M3 shared scales (1 per 16-element sub-block)
 *   32 bytes packed 4-bit E2M1  values (2 per byte)
 *   = 36 bytes per 64 elements
 */

/* Raw byte size for a tensor of n_el elements in the given 4-bit type.
   Returns -1 for non-MXFP4/NVFP types. */
long wubu_fp4_raw_size(int ggml_type, long n_elems);

/* Dequantize one full row (n_elems floats) from quantized data. */
void dequantize_row_mxfp4(const unsigned char *data, float *output, long n_elems);
void dequantize_row_nvfp4(const unsigned char *data, float *output, long n_elems);

#endif /* WUBU_DEQUANT_FP4_H */
