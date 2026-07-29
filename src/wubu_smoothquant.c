/*
 * wubu_smoothquant.c -- SmoothQuant activation outlier migration (doc 005).
 * Self-contained C11. See header.
 */
#include "wubu_smoothquant.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

int wubu_smoothquant_init(wubu_smoothquant_t *sq, const float *W, int M, int K,
                           const float *X_calib, int nbatch, float alpha) {
    if (!sq || !W || !X_calib || M < 1 || K < 1 || nbatch < 1) return -1;
    if (alpha < 0) alpha = 0; if (alpha > 1) alpha = 1;
    sq->M = M; sq->K = K; sq->alpha = alpha;
    sq->W_smooth = (float *)malloc((size_t)M * K * sizeof(float));
    sq->s = (float *)malloc(sizeof(float) * K);
    if (!sq->W_smooth || !sq->s) { wubu_smoothquant_free(sq); return -1; }

    /* per-channel max abs of activations over the calibration batch */
    float *maxX = (float *)malloc(sizeof(float) * K);
    float *maxW = (float *)malloc(sizeof(float) * K);
    if (!maxX || !maxW) { free(maxX); free(maxW); wubu_smoothquant_free(sq); return -1; }
    for (int k = 0; k < K; k++) { maxX[k] = 1e-12f; maxW[k] = 1e-12f; }
    for (int b = 0; b < nbatch; b++) {
        const float *xb = X_calib + (size_t)b * K;
        for (int k = 0; k < K; k++) { float a = fabsf(xb[k]); if (a > maxX[k]) maxX[k] = a; }
    }
    for (int k = 0; k < K; k++)
        for (int m = 0; m < M; m++) { float a = fabsf(W[(size_t)m*K+k]); if (a > maxW[k]) maxW[k] = a; }

    for (int k = 0; k < K; k++) {
        float sc = powf(maxX[k], alpha) / powf(maxW[k], 1.0f - alpha);
        if (!isfinite(sc) || sc <= 0) sc = 1.0f;
        sq->s[k] = sc;
    }
    /* W' = W · diag(s) */
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++)
            sq->W_smooth[(size_t)m*K+k] = W[(size_t)m*K+k] * sq->s[k];

    free(maxX); free(maxW);
    return 0;
}

void wubu_smoothquant_activate(const wubu_smoothquant_t *sq, const float *x_in, float *x_out) {
    for (int k = 0; k < sq->K; k++) x_out[k] = x_in[k] / sq->s[k];
}

void wubu_smoothquant_gemv(const wubu_smoothquant_t *sq, const float *x_smoothed, float *y) {
    for (int m = 0; m < sq->M; m++) {
        const float *wr = sq->W_smooth + (size_t)m * sq->K;
        float acc = 0.0f;
        for (int k = 0; k < sq->K; k++) acc += wr[k] * x_smoothed[k];
        y[m] = acc;
    }
}

void wubu_smoothquant_free(wubu_smoothquant_t *sq) {
    if (!sq) return;
    if (sq->W_smooth) free(sq->W_smooth);
    if (sq->s) free(sq->s);
    sq->W_smooth = NULL; sq->s = NULL;
}
