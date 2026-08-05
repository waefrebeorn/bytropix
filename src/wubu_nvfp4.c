/*
 * wubu_nvfp4.c — NVFP4 (E2M1 + microscaling) emulation (doc B08). C11.
 */
#include "wubu_nvfp4.h"
#include "wubu_fp8.h"   /* reuse E4M3 scale packing */
#include <math.h>
#include <stdlib.h>

uint8_t wubu_nvfp4_from_f32(float x) {
    if (x == 0.0f) return 0x0;
    int sign = (x < 0.0f) ? 0x8 : 0x0;
    float ax = fabsf(x);
    if (ax > 6.0f) ax = 6.0f;
    int exp; float m = frexpf(ax, &exp);  /* m in [0.5,1) */
    int e2m1 = exp;                       /* E2M1 bias = 1: value = (1+mant/2)*2^(e-1) */
    if (e2m1 <= 0) return (uint8_t)sign;  /* below subnormal -> 0 */
    if (e2m1 > 3) e2m1 = 3;               /* cap exponent */
    int mant = (int)((m * 2.0f - 1.0f) * 2.0f + 0.5f); /* 0 or 1 */
    if (mant >= 2) { mant = 0; e2m1++; if (e2m1 > 3) { e2m1 = 3; mant = 1; } }
    return (uint8_t)(sign | (e2m1 << 1) | (mant & 0x1));
}

float wubu_nvfp4_to_f32(uint8_t b) {
    if (b == 0) return 0.0f;
    int sign = (b & 0x8) ? -1 : 1;
    int exp = (b >> 1) & 0x3;
    int mant = b & 0x1;
    float unit = (float)ldexp(1.0f, exp - 1);  /* exp=0 -> 0.5 (subnorm) */
    return sign * (1.0f + mant * 0.5f) * unit;
}

int wubu_nvfp4_block_quantize(const float *x, uint8_t *packed,
                               uint8_t *scale_out, int n, int block) {
    if (block < 1) block = 16;
    int nb = (n + block - 1) / block;
    for (int bk = 0; bk < nb; bk++) {
        int start = bk * block;
        int cnt = (bk + 1) * block <= n ? block : n - start;
        float mx = 0.0f;
        for (int i = 0; i < cnt; i++) {
            float a = fabsf(x[start + i]);
            if (a > mx) mx = a;
        }
        float scale = mx / 6.0f;
        if (scale <= 0.0f) scale = 1e-6f;
        scale_out[bk] = wubu_fp8_e4m3_from_f32(scale);
        for (int i = 0; i < cnt; i++) {
            int byte = start / 2 + i / 2;   /* 2 elements per byte */
            int half = i & 1;                /* 0 = low nibble, 1 = high */
            uint8_t q = wubu_nvfp4_from_f32(x[start + i] / scale);
            if (half == 0) packed[byte] = q & 0xF;
            else packed[byte] |= (q & 0xF) << 4;
        }
    }
    return nb;
}

void wubu_nvfp4_gemv(const uint8_t *W, const uint8_t *scale, const float *A,
                      float *out, int rows, int n, int block) {
    if (block < 1) block = 16;
    for (int r = 0; r < rows; r++) {
        double s = 0.0;
        const uint8_t *wr = W + (size_t)r * n / 2;   /* 2 elements / byte */
        const uint8_t *sr = scale + (size_t)r * ((n + block - 1) / block);
        for (int i = 0; i < n; i++) {
            int blk = i / block;
            float sc = wubu_fp8_e4m3_to_f32(sr[blk]);
            uint8_t q = (i & 1) ? (wr[i / 2] >> 4) : (wr[i / 2] & 0xF);
            float w = wubu_nvfp4_to_f32(q) * sc;
            s += (double)w * (double)A[i];
        }
        out[r] = (float)s;
    }
}
