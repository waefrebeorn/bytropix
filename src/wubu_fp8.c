/*
 * wubu_fp8.c — FP8 E4M3 / E5M2 emulation (doc B07). Pure C11, no 3rd-party.
 */
#include "wubu_fp8.h"
#include <math.h>
#include <stdint.h>

/* ---- E4M3 (bias 7, 3 mantissa, no Inf; 1111 111 = NaN) ---- */
uint8_t wubu_fp8_e4m3_from_f32(float x) {
    if (x != x) return 0x7F;                 /* NaN */
    if (x == 0.0f) return 0x00;
    int sign = (x < 0.0f) ? 0x80 : 0x00;
    float ax = fabsf(x);
    if (ax > 448.0f) ax = 448.0f;           /* saturate */
    int e; float m = frexpf(ax, &e);        /* m in [0.5,1) -> 1.mant = m*2 */
    int exp = e + 6;                        /* E4M3 bias = 7 */
    if (exp <= 0) {
        /* subnormal */
        int sh = 1 - exp;
        if (sh > 3) return (uint8_t)sign;
        uint32_t mant = (uint32_t)((m * 2.0f - 1.0f) * (1u << (3 - sh)) + 0.5f);
        if (mant > 0x07) mant = 0x07;
        return (uint8_t)(sign | (mant & 0x07));
    }
    if (exp >= 0x0F) return 0x7F;
    int mant = (int)((m * 2.0f - 1.0f) * 8.0f + 0.5f);
    if (mant >= 8) { mant = 0; exp++; }     /* round-to-nearest-even carry */
    if (exp >= 0x0F) return 0x7F;
    return (uint8_t)(sign | (exp << 3) | (mant & 0x07));
}

float wubu_fp8_e4m3_to_f32(uint8_t b) {
    int sign = (b & 0x80) ? -1 : 1;
    int exp = (b >> 3) & 0x0F;
    int mant = b & 0x07;
    if (exp == 0x0F) return NAN;            /* E4M3 has no Inf: 1111 -> NaN */
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        return sign * (mant / 8.0f) * 0.5f * 0.015625f; /* 2^(-7) * (mant/8) */
    }
    float m = 1.0f + mant / 8.0f;
    return sign * m * (float)ldexp(1.0, exp - 7);
}

uint8_t wubu_fp8_e5m2_from_f32(float x) {
    if (x != x) return 0x7F;                 /* NaN */
    if (x == 0.0f) return 0x00;
    int sign = (x < 0.0f) ? 0x80 : 0x00;
    float ax = fabsf(x);
    int e; float m = frexpf(ax, &e);
    int exp = e + 14;                       /* bias 15 */
    if (exp <= 0) {
        int sh = 1 - exp;
        if (sh > 2) return (uint8_t)sign;
        uint32_t mant = (uint32_t)((m * 2.0f - 1.0f) * (1u << (2 - sh)) + 0.5f);
        if (mant > 0x03) mant = 0x03;
        return (uint8_t)(sign | (mant & 0x03));
    }
    if (exp >= 0x1F) return (uint8_t)(sign | 0x1F);  /* Inf */
    int mant = (int)((m * 2.0f - 1.0f) * 4.0f + 0.5f);
    if (mant >= 4) { mant = 0; exp++; }     /* carry */
    if (exp >= 0x1F) return (uint8_t)(sign | 0x1F);
    return (uint8_t)(sign | (exp << 2) | (mant & 0x03));
}

float wubu_fp8_e5m2_to_f32(uint8_t b) {
    int sign = (b & 0x80) ? -1 : 1;
    int exp = (b >> 2) & 0x1F;
    int mant = b & 0x03;
    if (exp == 0x1F) return (b & 0x03) ? NAN : (sign * INFINITY);
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        return sign * (mant / 4.0f) * 0.5f * 0.03125f; /* 2^(-15) * (mant/4) */
    }
    float m = 1.0f + mant / 4.0f;
    return sign * m * (float)ldexp(1.0, exp - 15);
}

int wubu_fp8_quantize(const float *x, uint8_t *out, int n, int e5m2) {
    for (int i = 0; i < n; i++)
        out[i] = e5m2 ? wubu_fp8_e5m2_from_f32(x[i])
                      : wubu_fp8_e4m3_from_f32(x[i]);
    return n;
}

void wubu_fp8_dequantize(const uint8_t *q, float *out, int n, int e5m2) {
    for (int i = 0; i < n; i++)
        out[i] = e5m2 ? wubu_fp8_e5m2_to_f32(q[i])
                      : wubu_fp8_e4m3_to_f32(q[i]);
}

float wubu_fp8_dot(const uint8_t *w_fp8, const float *act, int n, int e5m2) {
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        float w = e5m2 ? wubu_fp8_e5m2_to_f32(w_fp8[i])
                       : wubu_fp8_e4m3_to_f32(w_fp8[i]);
        s += (double)w * (double)act[i];
    }
    return (float)s;
}

void wubu_fp8_gemv(const uint8_t *W, const float *A, float *out,
                   int rows, int n, int e5m2) {
    for (int r = 0; r < rows; r++)
        out[r] = wubu_fp8_dot(W + (size_t)r * n, A, n, e5m2);
}
