/*
 * wubu_ternary.c -- BitNet/ternary-weight kernels (T01/T02/T03). C11.
 *
 * Convergence (BitNet b1.58 / bitnet.cpp 7-hop):
 *   - T01 ternary pack: map F32 weights to {-1,0,+1} via absmax scaling
 *        (w_scaled = w / maxabs; ternary = sign(w_scaled) clipped to {-1,0,1}
 *        with a threshold). Pack 4 ternary values (2 bits each) into one byte.
 *   - T03 absmax scaling: return the scale = max|w| so dequant recovers
 *        w_hat = ternary * scale. (Round-trip fidelity check lives in test.)
 *   - T02 mpGEMV: y = sum_i ternary_w[i] * int8_act[i] * scale  (the matvec that
 *        replaces the F32 GEMV for ternary-weight layers -> ~3.9x less bandwidth).
 *
 * Triple-DA: n<=0 -> 0; null -> 0; scale<=0 guard; deterministic pack.
 */
#include "wubu_ternary.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* T03 absmax scaling factor for a weight tensor. */
float wubu_ternary_scale(const float *w, int n) {
    if (!w || n <= 0) return 0.0f;
    float m = 0.0f;
    for (int i = 0; i < n; i++) { float a = fabsf(w[i]); if (a > m) m = a; }
    return m > 0.0f ? m : 1.0f;
}

/* T01 pack: ternaryize w (threshold thr, default 0.5 of scale) and pack 4/byte.
 * out byte array sized >= (n+3)/4. Returns bytes written. */
int wubu_ternary_pack(const float *w, int n, float scale, float thr,
                      unsigned char *out) {
    if (!w || !out || n <= 0 || scale <= 0.0f) return 0;
    if (thr < 0.0f) thr = 0.5f;
    for (int i = 0; i < n; i += 4) {
        unsigned char b = 0;
        for (int t = 0; t < 4 && i+t < n; t++) {
            float s = w[i+t] / scale;
            int tv = (s >  thr) ? 1 : (s < -thr) ? -1 : 0;  /* {-1,0,1} */
            tv &= 3; /* 2-bit field; -1 -> 3 which we map back to -1 on dequant */
            b |= ((unsigned char)(tv & 3)) << (2*t);
        }
        out[i/4] = b;
    }
    return (n + 3) / 4;
}

/* dequant a packed byte array back to F32 (inverse of pack). */
int wubu_ternary_unpack(const unsigned char *packed, int n, float scale,
                        float *out) {
    if (!packed || !out || n <= 0 || scale <= 0.0f) return 0;
    for (int i = 0; i < n; i++) {
        unsigned char b = packed[i/4];
        int tv = (b >> (2*(i%4))) & 3;
        float v = (tv == 3) ? -1.0f : (float)tv;  /* 3 -> -1, else 0/1 */
        out[i] = v * scale;
    }
    return n;
}

/* T02 mpGEMV: y[o] = scale * sum_i ternary_w[i + o*cols] * act[i]  (int8-ish act
 * passed as float; ternary_w is a pre-packed byte array). Returns rows computed. */
int wubu_mpgemv(const unsigned char *tw, int rows, int cols, float scale,
                const float *act, float *y) {
    if (!tw || !act || !y || rows <= 0 || cols <= 0 || scale <= 0.0f) return 0;
    for (int o = 0; o < rows; o++) {
        float acc = 0.0f;
        for (int i = 0; i < cols; i++) {
            int idx = o*cols + i;
            unsigned char b = tw[idx/4];
            int tv = (b >> (2*(idx%4))) & 3;
            float w = (tv == 3) ? -1.0f : (float)tv;
            acc += w * act[i];
        }
        y[o] = scale * acc;
    }
    return rows;
}
