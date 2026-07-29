/*
 * wubu_ternary.c -- BitNet 1.58 ternary {-1,0,+1} GEMV (doc 004).
 * Self-contained C11. See header.
 */
#include "wubu_ternary.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* value mapping: 0->-1, 1->0, 2->+1, 3->pad(0) */
static int8_t decode_val(int v) {
    switch (v & 3) {
        case 0: return -1;
        case 1: return  0;
        case 2: return +1;
        default: return 0; /* pad */
    }
}
static int enc_val(int8_t w) {
    if (w < 0) return 0;   /* -1 -> 0 */
    if (w > 0) return 2;   /* +1 -> 2 */
    return 1;              /*  0 -> 1 */
}

int wubu_ternary_packed_bytes(int K) { return (K + 3) / 4; }

void wubu_ternary_pack_row(const int8_t *wq, int K, uint8_t *out) {
    int n = wubu_ternary_packed_bytes(K);
    memset(out, 0, n);
    for (int i = 0; i < K; i++) {
        int v = enc_val(wq[i]);
        int nib = i & 3;            /* 0..3 within a byte */
        int byte = i >> 2;
        out[byte] |= (uint8_t)(v << (2 * (3 - nib)));  /* MSB-first */
    }
}
void wubu_ternary_unpack_row(const uint8_t *buf, int K, int8_t *wq_out) {
    for (int i = 0; i < K; i++) {
        int nib = i & 3;
        int byte = i >> 2;
        int v = (buf[byte] >> (2 * (3 - nib))) & 3;
        wq_out[i] = decode_val(v);
    }
}

int wubu_ternary_quantize(const float *W, int M, int K, wubu_ternary_t *q) {
    if (!q || !W || M < 1 || K < 1) return -1;
    q->M = M; q->K = K; q->K_packed = wubu_ternary_packed_bytes(K);
    q->t = (int8_t *)malloc((size_t)M * q->K_packed);
    q->scale = (float *)malloc(sizeof(float) * M);
    if (!q->t || !q->scale) { wubu_ternary_free(q); return -1; }

    int8_t *wrk = (int8_t *)malloc(K * sizeof(int8_t));
    if (!wrk) { wubu_ternary_free(q); return -1; }

    for (int m = 0; m < M; m++) {
        const float *Wr = W + (size_t)m * K;
        /* per-row absmean scale */
        double s = 0; for (int k = 0; k < K; k++) s += fabsf(Wr[k]);
        float scale = (float)(s / K) + 1e-12f;
        q->scale[m] = scale;
        for (int k = 0; k < K; k++) {
            float qf = Wr[k] / scale;
            int v = (int)(qf >= 0 ? qf + 0.5f : qf - 0.5f);
            if (v >  1) v =  1;
            if (v < -1) v = -1;
            wrk[k] = (int8_t)v;
        }
        wubu_ternary_pack_row(wrk, K, q->t + (size_t)m * q->K_packed);
    }
    free(wrk);
    return 0;
}

void wubu_ternary_gemv(const wubu_ternary_t *q, const float *x, float *y) {
    for (int m = 0; m < q->M; m++) {
        const uint8_t *tr = q->t + (size_t)m * q->K_packed;
        float acc = 0.0f;
        int K = q->K;
        for (int i = 0; i < K; i++) {
            int nib = i & 3;
            int byte = i >> 2;
            int v = (tr[byte] >> (2 * (3 - nib))) & 3;
            int8_t w = decode_val(v);
            if (w) acc += w * x[i];
        }
        y[m] = acc * q->scale[m];
    }
}

void wubu_ternary_free(wubu_ternary_t *q) {
    if (!q) return;
    if (q->t) free(q->t);
    if (q->scale) free(q->scale);
    q->t = NULL; q->scale = NULL;
}
