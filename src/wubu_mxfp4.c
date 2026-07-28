/*
 * wubu_mxfp4.c — OCP Microscaling (MX) FP4/FP8 quant (Round-4 #433/#435/#437/#438).
 * C11, self-contained. MXFP4: 32-element blocks, each element E2M1 (1s/2e/1m),
 * one shared E8M0 (8-bit pure-exponent, power-of-2) scale per block. MXFP8:
 * E4M3 elements (1s/4e/3m) + same E8M0 block scale. Dequant: value = scale * e.
 * This is the format of Kimi K3's 594GB release weights (MXFP4) and activations
 * (MXFP8, SiTU). E8M0 scale = 2^(int8 as signed-ish exponent); we store the
 * exponent byte directly (0..255 => 2^(e-127) bias, matching OCP E8M0).
 */
#include "wubu_mxfp4.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define MX_BLOCK 32

/* E2M1 (FP4) encode: representable vals {0, 0.5, 1, 1.5, 2, 3, 4, 6} * sign.
 * We map a normalized magnitude m in [0,6] to nearest E2M1 level. */
static int e2m1_encode(float v) {
    int sign = (v < 0) ? 1 : 0;
    float a = fabsf(v);
    /* E2M1 magnitude levels: 0, 0.5, 1, 1.5, 2, 3, 4, 6 */
    float levels[8] = {0, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    int best = 0; float bd = 1e30f;
    for (int i = 0; i < 8; i++) { float d = fabsf(a - levels[i]); if (d < bd) { bd = d; best = i; } }
    return (sign << 3) | best;   /* 4-bit: [sign|exp2|mant1] */
}
static float e2m1_decode(int code) {
    float levels[8] = {0, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float v = levels[code & 0x7];
    return (code & 0x8) ? -v : v;
}

/* E4M3 (FP8) encode/decode (1s/4e/3m, bias 7, max normal ~448, subnormals).
 * Proper float32 -> E4M3 bit conversion with mantissa rounding. */
static int e4m3_encode(float v) {
    if (v == 0.0f) return 0;
    uint32_t u; memcpy(&u, &v, sizeof(u));
    int sign = (u >> 31) & 1;
    int exp  = (int)((u >> 23) & 0xFF);
    int mant = (int)(u & 0x7FFFFF);
    if (exp == 0xFF) return sign ? 0x80 : 0x7F;   /* inf/nan -> max */
    int e = exp - 127 + 7;                          /* rebias to E4M3 (bias 7) */
    int m = mant >> (23 - 3);                       /* keep top 3 mantissa bits */
    int lsb = (mant >> (23 - 3 - 1)) & 1;           /* rounding bit */
    if (lsb) { m++; if (m > 7) { m = 0; e++; } }    /* round half-up, renormalize */
    if (e <= 0) {                                   /* subnormal or underflow -> 0 */
        if (e < -3) return sign ? 0x80 : 0x00;
        m = (m | 0x8) >> (1 - e);                   /* shift into subnormal */
        e = 0;
    }
    /* OCP E4M3: NO inf/NaN. Exponent field 15 is a VALID normal (max 448),
     * so only clamp true overflow (e > 15) to the max representable (exp=15,mant=6). */
    if (e > 15) { e = 15; m = 6; }
    if (e < 0) e = 0;
    return (sign << 7) | (e << 3) | (m & 0x7);
}
static float e4m3_decode(int code) {
    int sign = (code >> 7) & 1;
    int e = (code >> 3) & 0xF;
    int m = code & 0x7;
    if (e == 0) {                                   /* subnormal */
        if (m == 0) return sign ? -0.0f : 0.0f;
        float v = (float)m / 8.0f;                 /* 0.mmm * 2^-6 */
        return sign ? -v : v;
    }
    /* OCP E4M3: all exp fields 1..15 are normals (incl. e==15 = max 448). */
    float v = (1.0f + (float)m / 8.0f) * exp2f((float)(e - 7));
    return sign ? -v : v;
}

/* Pack `n` floats into MXFP4 (k=32 blocks): out = n/32 bytes (4 bits/elm) +
 * n/32 scale bytes (E8M0). scale = 2^(exp) chosen so max|v| maps to level 6 (6.0).
 * Returns bytes written, or -1 on bad args. */
int wubu_mxfp4_pack(const float *x, int n, uint8_t *out) {
    if (n % MX_BLOCK != 0 || !x || !out) return -1;
    int nblk = n / MX_BLOCK;
    for (int b = 0; b < nblk; b++) {
        const float *xb = x + b * MX_BLOCK;
        float amax = 0;
        for (int i = 0; i < MX_BLOCK; i++) { float a = fabsf(xb[i]); if (a > amax) amax = a; }
        if (amax < 1e-12f) amax = 1e-12f;
        /* scale s.t. 6.0 * 2^E >= amax  => E = ceil(log2(amax/6)) */
        float Ef = ceilf(log2f(amax / 6.0f));
        int E = (int)Ef;
        if (E < -127) E = -127;
        if (E > 127) E = 127;
        uint8_t scale = (uint8_t)(E + 127);     /* E8M0 bias 127 */
        float scale_f = exp2f((float)E);
        uint8_t *blk = out + b * (MX_BLOCK/2 + 1);
        blk[MX_BLOCK/2] = scale;
        for (int i = 0; i < MX_BLOCK; i++) {
            float q = xb[i] / scale_f;            /* de-normalize to E2M1 range */
            int code = e2m1_encode(q);
            if (i % 2 == 0) blk[i/2] = (uint8_t)code;
            else blk[i/2] |= (uint8_t)(code << 4);
        }
    }
    return nblk * (MX_BLOCK/2 + 1);
}

/* Unpack MXFP4 (k=32) back to floats. Returns 0 on ok. */
int wubu_mxfp4_unpack(const uint8_t *in, int n, float *out) {
    if (n % MX_BLOCK != 0 || !in || !out) return -1;
    int nblk = n / MX_BLOCK;
    for (int b = 0; b < nblk; b++) {
        const uint8_t *blk = in + b * (MX_BLOCK/2 + 1);
        int E = (int)blk[MX_BLOCK/2] - 127;
        float scale_f = exp2f((float)E);
        for (int i = 0; i < MX_BLOCK; i++) {
            int code = (i % 2 == 0) ? (blk[i/2] & 0xF) : (blk[i/2] >> 4);
            out[b*MX_BLOCK + i] = e2m1_decode(code) * scale_f;
        }
    }
    return 0;
}

/* Pack/unpack MXFP8 (E4M3). Layout: n bytes + nblk scale bytes. */
int wubu_mxfp8_pack(const float *x, int n, uint8_t *out) {
    if (n % MX_BLOCK != 0 || !x || !out) return -1;
    int nblk = n / MX_BLOCK;
    for (int b = 0; b < nblk; b++) {
        const float *xb = x + b * MX_BLOCK;
        float amax = 0;
        for (int i = 0; i < MX_BLOCK; i++) { float a = fabsf(xb[i]); if (a > amax) amax = a; }
        if (amax < 1e-12f) amax = 1e-12f;
        float Ef = ceilf(log2f(amax / 448.0f));
        int E = (int)Ef;
        if (E < -127) E = -127;
        if (E > 127) E = 127;
        uint8_t scale = (uint8_t)(E + 127);
        float scale_f = exp2f((float)E);
        out[b*(MX_BLOCK+1) + MX_BLOCK] = scale;
        for (int i = 0; i < MX_BLOCK; i++)
            out[b*(MX_BLOCK+1) + i] = (uint8_t)e4m3_encode(xb[i] / scale_f);
    }
    return nblk * (MX_BLOCK + 1);
}
int wubu_mxfp8_unpack(const uint8_t *in, int n, float *out) {
    if (n % MX_BLOCK != 0 || !in || !out) return -1;
    int nblk = n / MX_BLOCK;
    for (int b = 0; b < nblk; b++) {
        int E = (int)in[b*(MX_BLOCK+1) + MX_BLOCK] - 127;
        float scale_f = exp2f((float)E);
        for (int i = 0; i < MX_BLOCK; i++)
            out[b*MX_BLOCK + i] = e4m3_decode(in[b*(MX_BLOCK+1)+i]) * scale_f;
    }
    return 0;
}
