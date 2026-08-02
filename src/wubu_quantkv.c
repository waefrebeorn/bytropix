/*
 * wubu_quantkv.c -- KV cache quantization (INT8 group-wise) (HH06). C11.
 *
 * Convergence (INT8/FP8 KV quantization / GGUF q4 7-hop):
 *   - HH06: KV cache is memory-bound at 512K ctx. INT8 per-group (symmetric)
 *     quantization quarters memory vs FP32 (4x compression), 2x vs FP16.
 *     Per-group scales (like QLoRA/GGUF q4) preserve attention quality. At home:
 *     wubu_quantkv quantizes the 512K KV to INT8 → 2x (vs FP16) more ctx
 *     headroom without EAMM, directly serving the 512K + 27 tok/s mandate.
 */
#include "wubu_quantkv.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

int wubu_quantkv_bits(void) { return 8; }
float wubu_quantkv_ratio(void) { return 32.0f / 8.0f; }  /* FP32 → INT8 = 4x */

int wubu_quantkv_quantize(wubu_quantkv_t *qk, const float *kv, int n) {
    if (!qk || !kv || n <= 0 || n > WUBU_QUANTKV_MAX) return -1;
    if (qk->group <= 0) qk->group = WUBU_QUANTKV_GROUP;
    qk->n = n;
    qk->zero = 0.0f;
    int ng = (n + qk->group - 1) / qk->group;
    for (int g = 0; g < ng; g++) {
        int start = g * qk->group;
        int end = (start + qk->group < n) ? start + qk->group : n;
        float max_abs = 1e-12f;
        for (int i = start; i < end; i++) {
            float a = fabsf(kv[i]);
            if (a > max_abs) max_abs = a;
        }
        qk->scale[g] = max_abs / 127.0f;   /* symmetric: [-127,127] → [-max,max] */
        for (int i = start; i < end; i++) {
            int v = (int)lroundf(kv[i] / qk->scale[g]);
            if (v > 127) v = 127; if (v < -127) v = -127;
            qk->q[i] = (signed char)v;
        }
    }
    return 0;
}

int wubu_quantkv_dequantize(const wubu_quantkv_t *qk, float *out) {
    if (!qk || !out) return -1;
    int ng = (qk->n + qk->group - 1) / qk->group;
    (void)ng;
    for (int i = 0; i < qk->n; i++) {
        int g = i / qk->group;
        out[i] = (float)qk->q[i] * qk->scale[g];
    }
    return 0;
}
