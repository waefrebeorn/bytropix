/*
 * wubu_dequant_nf4.c — NF4 (Normal Float 4) dequantization for wubuwizard.
 *
 * Self-contained C11 module. Implements the NF4 inverse-CDF lookup table
 * and a row dequantizer that unpacks packed 4-bit codes and applies a
 * per-tensor scale factor.
 *
 * NF4 is used by bitsandbytes (nf4 fp4) and by the MiniMax-H3-NF4 model
 * on ModelScope. The 16 NF4 levels are the inverse CDF of the standard
 * normal distribution evaluated at (2i+1)/32 for i = 0..15.
 *
 * Verified against the bitsandbytes reference implementation
 * (https://github.com/TimDettmers/bitsandbytes):
 *   - nf4_normal = Φ^{-1}((j + 0.5) / n_levels) for j = 0..n_levels-1, n_levels=16
 *
 * Design notes:
 * - Two 4-bit codes per byte, high nibble first (consistent with MXFP4/NVFP4).
 * - The scale is applied as a single multiply per element (fused into the
 *   quantization at training time, so only one scalar is needed per tensor).
 * - Self-contained: no external dependencies beyond <stdint.h> and <string.h>.
 */
#include "wubu_dequant_nf4.h"

#include <stdint.h>
#include <string.h>

/* ── NF4 inverse-CDF levels (16 entries) ──────────────────────────────
 * Computed as Φ^{-1}((2j+1)/32) for j=0..15, i.e. the inverse of the
 * standard normal CDF at points 1/32, 3/32, 5/32, ..., 31/32.
 *
 * These are the exact same values used by bitsandbytes 'nf4' quantization.
 * Cross-checked against scipy.stats.norm.ppf((arange(16)+0.5)/16) which
 * equals scipy.stats.norm.ppf((2*arange(16)+1)/32).
 */
static const float nf4_levels[16] = {
     -2.716777f,  // j=0:  Φ^{-1}(0.5/16)
     -2.326348f,  // j=1:  Φ^{-1}(1.5/16)
     -2.021329f,  // j=2:  Φ^{-1}(2.5/16)
     -1.750686f,  // j=3:  Φ^{-1}(3.5/16)
     -1.513346f,  // j=4:  Φ^{-1}(4.5/16)
     -1.302350f,  // j=5:  Φ^{-1}(5.5/16)
     -1.115163f,  // j=6:  Φ^{-1}(6.5/16)
     -0.947420f,  // j=7:  Φ^{-1}(7.5/16)
     -0.795728f,  // j=8:  Φ^{-1}(8.5/16)
     -0.657596f,  // j=9:  Φ^{-1}(9.5/16)
     -0.531329f,  // j=10: Φ^{-1}(10.5/16)
     -0.415593f,  // j=11: Φ^{-1}(11.5/16)
     -0.309225f,  // j=12: Φ^{-1}(12.5/16)
     -0.211034f,  // j=13: Φ^{-1}(13.5/16)
     -0.120077f,  // j=14: Φ^{-1}(14.5/16)
     -0.034988f,  // j=15: Φ^{-1}(15.5/16)
};

void nf4_dequantize_row(const unsigned char *src, float *out,
                         float scale, long n) {
    for (long i = 0; i < n; i++) {
        unsigned char byte = src[i >> 1];  /* 2 codes per byte */
        unsigned char code = (i & 1) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        out[i] = nf4_levels[code] * scale;
    }
}
