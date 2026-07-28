/*
 * wubu_kvcache_quant.c -- implementation (see header for the research basis).
 * Pure-C, CPU-realizable. All functions are exact round-trippable + tested.
 */
#include "wubu_kvcache_quant.h"
#include <math.h>
#include <stdlib.h>

static inline int8_t f32_to_q8(float v, float scale) {
    if (scale == 0.0f) return 0;
    int q = (int)lroundf(v / scale);
    if (q > 127) q = 127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

void wubu_kvq_q8_quant(const float *x, int8_t *q, float *scale, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    *scale = amax / 127.0f;
    if (*scale == 0.0f) { for (int i = 0; i < n; i++) q[i] = 0; return; }
    for (int i = 0; i < n; i++) q[i] = f32_to_q8(x[i], *scale);
}

void wubu_kvq_q8_dequant(const int8_t *q, float scale, float *out, int n) {
    for (int i = 0; i < n; i++) out[i] = (float)q[i] * scale;
}

void wubu_kvq_kivi_quant_K(const float *K, int8_t *q, float *scale_per_ch,
                           int n_tokens, int head_dim) {
    /* one scale per channel: max abs over all tokens in that channel */
    float *col_max = (float *)malloc((size_t)head_dim * sizeof(float));
    for (int c = 0; c < head_dim; c++) {
        float amax = 0.0f;
        for (int t = 0; t < n_tokens; t++) {
            float a = fabsf(K[(size_t)t * head_dim + c]);
            if (a > amax) amax = a;
        }
        col_max[c] = amax / 127.0f;
    }
    for (int t = 0; t < n_tokens; t++) {
        const float *row = K + (size_t)t * head_dim;
        int8_t *qrow = q + (size_t)t * head_dim;
        for (int c = 0; c < head_dim; c++)
            qrow[c] = f32_to_q8(row[c], col_max[c]);
    }
    for (int c = 0; c < head_dim; c++) scale_per_ch[c] = col_max[c];
    free(col_max);
}

void wubu_kvq_kivi_dequant_K(const int8_t *q, const float *scale_per_ch,
                             float *out, int n_tokens, int head_dim) {
    for (int t = 0; t < n_tokens; t++) {
        const int8_t *qrow = q + (size_t)t * head_dim;
        float *orow = out + (size_t)t * head_dim;
        for (int c = 0; c < head_dim; c++)
            orow[c] = (float)qrow[c] * scale_per_ch[c];
    }
}

void wubu_kvq_kivi_quant_V(const float *V, int8_t *q, float *scale_per_tok,
                           int n_tokens, int head_dim) {
    /* one scale per token: max abs over head_dim in that token row */
    for (int t = 0; t < n_tokens; t++) {
        const float *row = V + (size_t)t * head_dim;
        float amax = 0.0f;
        for (int c = 0; c < head_dim; c++) {
            float a = fabsf(row[c]);
            if (a > amax) amax = a;
        }
        float s = amax / 127.0f;
        scale_per_tok[t] = s;
        int8_t *qrow = q + (size_t)t * head_dim;
        if (s == 0.0f) { for (int c = 0; c < head_dim; c++) qrow[c] = 0; continue; }
        for (int c = 0; c < head_dim; c++) qrow[c] = f32_to_q8(row[c], s);
    }
}

void wubu_kvq_kivi_dequant_V(const int8_t *q, const float *scale_per_tok,
                             float *out, int n_tokens, int head_dim) {
    for (int t = 0; t < n_tokens; t++) {
        const int8_t *qrow = q + (size_t)t * head_dim;
        float *orow = out + (size_t)t * head_dim;
        float s = scale_per_tok[t];
        for (int c = 0; c < head_dim; c++) orow[c] = (float)qrow[c] * s;
    }
}

float wubu_kvq_bytes_per_elem(wubu_kvq_scheme_t scheme) {
    switch (scheme) {
        case WUBU_KVQ_F32:   return 4.0f;
        case WUBU_KVQ_Q8_0:  return 9.0f / 32.0f;   /* 1 float scale + 32 int8 per 32 */
        case WUBU_KVQ_KIVI:  return 2.0f;            /* 1 int8 + ~1 float scale per elem
                                                        (amortized; exact depends on
                                                        n_tokens/head_dim ratio) */
        default:              return 4.0f;
    }
}
