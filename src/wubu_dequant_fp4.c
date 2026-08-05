/*
 * wubu_dequant_fp4.c — C11 row-level dequantization for OCP MXFP4 and
 * NVFP4 microscaling 4-bit formats.  Self-contained: only stdlib, math,
 * string.  No third-party deps.
 *
 * MXFP4 (ggml type 39, QK_MXFP4 = 32):
 *   struct { uint8_t e; uint8_t qs[16]; }
 *   e  = E8M0 shared exponent byte (bias 127)
 *   qs = 16 bytes, each holding 2 × E2M1 nibbles
 *
 * NVFP4 (ggml type 40, QK_NVFP4 = 64, sub = 16):
 *   struct { uint8_t d[4]; uint8_t qs[32]; }
 *   d  = 4 × UE4M3 scale bytes (one per 16-element sub-block)
 *   qs = 32 bytes, each holding 2 × E2M1 nibbles
 *
 * E2M1 codeword → value:
 *   0 → 0, 1 → 0.5, 2 → 1.0, 3 → 2.0
 *
 * E8M0 (8-bit, bias 127): scale = 2^(e - 127)
 *
 * UE4M3 (unsigned 8-bit, 5 exp + 3 mant, bias 7):
 *   e == 0 && m == 0  → 0
 *   e == 0 && m != 0  → subnormal m * 2^(-6)
 *   0 < e < 31        → (1 + m/8) * 2^(e-7)
 *   e == 31           → Inf (treated as 0 here — shouldn't occur in practice)
 */

#include "wubu_dequant_fp4.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* E2M1 lookup table                                                  */
/* ------------------------------------------------------------------ */
static const float e2m1_val[4] = { 0.0f, 0.5f, 1.0f, 2.0f };

/* ------------------------------------------------------------------ */
/* E8M0 shared-scale decode (OCP MX standard)                          */
/* ------------------------------------------------------------------ */
static inline float e8m0_to_scale(uint8_t byte)
{
    /* bias = 127, range: 2^(-127) to 2^(128) */
    return powf(2.0f, (float)byte - 127.0f);
}

/* ------------------------------------------------------------------ */
/* UE4M3 shared-scale decode (NVFP4 / NVIDIA variant)                  */
/*  Unsigned E4M3: 5 exp bits (bias 7) + 3 mant bits, no sign.         */
/* ------------------------------------------------------------------ */
static inline float ue4m3_to_scale(uint8_t byte)
{
    int exp = (byte >> 3) & 0x1F;
    int mant = byte & 0x07;

    if (exp == 0) {
        if (mant == 0) return 0.0f;
        /* subnormal: m * 2^(-6) */
        return (float)mant * (1.0f / 64.0f);
    }
    if (exp == 31) {
        /* Inf/NaN — shouldn't happen in NVFP4, treat as 0 */
        return 0.0f;
    }
    /* normal: (1 + m/8) * 2^(exp-7) */
    return (1.0f + (float)mant * (1.0f / 8.0f)) * powf(2.0f, (float)exp - 7.0f);
}

/* ------------------------------------------------------------------ */
/* Raw byte size                                                      */
/* ------------------------------------------------------------------ */
long wubu_fp4_raw_size(int ggml_type, long n_elems)
{
    const long BLK = (ggml_type == 39) ? 32 : 64;   /* 39=MXFP4, 40=NVFP4 */
    const long BPB = (ggml_type == 39) ? 17 : 36;   /* bytes per block  */
    long nblocks = (n_elems + BLK - 1) / BLK;
    return nblocks * BPB;
}

/* ------------------------------------------------------------------ */
/* MXFP4 row dequant                                                   */
/* ------------------------------------------------------------------ */
void dequantize_row_mxfp4(const unsigned char *data, float *output, long n_elems)
{
    long i = 0;
    while (i < n_elems) {
        float scale = e8m0_to_scale(data[0]);
        const unsigned char *qs = data + 1;
        long remaining = n_elems - i;
        long to_do = remaining < 32 ? remaining : 32;

        for (long j = 0; j < to_do; j++) {
            unsigned char byte = qs[j >> 1];
            int nibble;
            if (j & 1)
                nibble = byte & 0x0F;
            else
                nibble = byte >> 4;
            output[i + j] = scale * e2m1_val[nibble];
        }

        data += 17;  /* 1 scale + 16 data bytes */
        i += 32;
    }
}

/* ------------------------------------------------------------------ */
/* NVFP4 row dequant                                                   */
/* ------------------------------------------------------------------ */
void dequantize_row_nvfp4(const unsigned char *data, float *output, long n_elems)
{
    long i = 0;
    while (i < n_elems) {
        const unsigned char *ds = data;          /* 4 UE4M3 scale bytes */
        const unsigned char *qs = data + 4;      /* 32 bytes: 64 × E2M1 */
        long remaining = n_elems - i;
        long to_do = remaining < 64 ? remaining : 64;

        for (long j = 0; j < to_do; j++) {
            long sub_idx = j / 16;                /* which 16-element sub-block */
            float scale = ue4m3_to_scale(ds[sub_idx]);
            unsigned char byte = qs[j >> 1];
            int nibble;
            if (j & 1)
                nibble = byte & 0x0F;
            else
                nibble = byte >> 4;
            output[i + j] = scale * e2m1_val[nibble];
        }

        data += 36;  /* 4 scales + 32 data bytes */
        i += 64;
    }
}
