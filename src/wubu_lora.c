/*
 * wubu_lora.c -- LoRA application (C11, self-contained, opaque).
 *
 * Merged form:  W' = W + scale * (B^T @ A),  scale = alpha/rank.
 *   A: [rank, in_f]   down
 *   B: [out_f, rank] up
 * B^T @ A has shape [out_f, in_f] (matches W). Applied in place.
 *
 * Forward form (frozen base):  y = x @ W^T + scale * ((x @ A^T) @ B^T)
 *   x @ A^T -> [rank];  (x@A^T) @ B^T -> [out_f].
 */

#include "wubu_lora.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct wubu_lora {
    int rank;
    int in_f;
    int out_f;
    float alpha;
    float scale;
    float *A;   // [rank * in_f]
    float *B;   // [out_f * rank]
};

wubu_lora_t *wubu_lora_create(int rank, float alpha, int in_f, int out_f) {
    if (rank <= 0 || in_f <= 0 || out_f <= 0) return NULL;
    wubu_lora_t *l = (wubu_lora_t *)calloc(1, sizeof(*l));
    if (!l) return NULL;
    l->rank = rank;
    l->in_f = in_f;
    l->out_f = out_f;
    l->alpha = alpha > 0.0f ? alpha : (float)rank;
    l->scale = l->alpha / (float)rank;
    l->A = (float *)malloc((size_t)rank * (size_t)in_f * sizeof(float));
    l->B = (float *)malloc((size_t)out_f * (size_t)rank * sizeof(float));
    if (!l->A || !l->B) { wubu_lora_free(l); return NULL; }
    return l;
}

void wubu_lora_free(wubu_lora_t *l) {
    if (!l) return;
    if (l->A) free(l->A);
    if (l->B) free(l->B);
    free(l);
}

int wubu_lora_load_f32(wubu_lora_t *l, const float *A, const float *B) {
    if (!l || !A || !B) return -1;
    memcpy(l->A, A, (size_t)l->rank * l->in_f * sizeof(float));
    memcpy(l->B, B, (size_t)l->out_f * l->rank * sizeof(float));
    return 0;
}

int wubu_lora_load_raw(wubu_lora_t *l, const float *A, const float *B) {
    return wubu_lora_load_f32(l, A, B);  // both f32
}

// accum[o*in_f + i] += scale * sum_r B[o*rank + r] * A[r*in_f + i]
static void lora_delta(const wubu_lora_t *l, float *delta) {
    int r = l->rank, inf = l->in_f, outf = l->out_f;
    for (int o = 0; o < outf; o++) {
        const float *Brow = l->B + (size_t)o * r;
        for (int i = 0; i < inf; i++) {
            float s = 0.0f;
            for (int rr = 0; rr < r; rr++)
                s += Brow[rr] * l->A[(size_t)rr * inf + i];
            delta[(size_t)o * inf + i] += l->scale * s;
        }
    }
}

int wubu_lora_apply(const wubu_lora_t *l, float *W) {
    if (!l || !W) return -1;
    lora_delta(l, W);  // W already has base; delta added in place
    return 0;
}

int wubu_lora_forward(const wubu_lora_t *l, const float *x, float *out) {
    if (!l || !x || !out) return -1;
    int r = l->rank, inf = l->in_f, outf = l->out_f;
    float *h = (float *)malloc((size_t)r * sizeof(float));
    if (!h) return -1;
    for (int rr = 0; rr < r; rr++) {
        float s = 0.0f;
        for (int i = 0; i < inf; i++) s += x[i] * l->A[(size_t)rr * inf + i];
        h[rr] = s;
    }
    for (int o = 0; o < outf; o++) {
        float s = 0.0f;
        for (int rr = 0; rr < r; rr++) s += h[rr] * l->B[(size_t)o * r + rr];
        out[o] = l->scale * s;
    }
    free(h);
    return 0;
}

float wubu_lora_scale(const wubu_lora_t *l) {
    return l ? l->scale : 0.0f;
}
