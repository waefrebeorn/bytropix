/*
 * wubu_smoothquant.h -- SmoothQuant activation outlier migration (doc 005).
 *
 * WHY (Kevin-Bacon convergence): int8/int4 GEMV (B01/B02/B03) quantizes each
 * row/channel with per-channel absmax. Activations of LLMs have a few massive
 * outlier channels (up to 100x the median) that inflate the activation scale and
 * crush the other 99% into rounding noise. SmoothQuant (Xiao et al., 2022)
 * migrates that outlier magnitude from the activation into the weight:
 *     s[c] = (max_b|x[c]|)^alpha / (max|W[:,c]|)^(1-alpha)
 *     W' = W · diag(s),   X' = diag(1/s) · X
 * THEN  W' @ X' == W @ X  EXACTLY (the s and 1/s cancel), but now BOTH W' and
 * X' have bounded per-channel ranges, so int8/int4 quantizes them cleanly.
 *
 * SCHEME (own-C): given weight W[M,K] and a calibration set of activations
 * X[nbatch, K], compute s[K], return W' and a function to smooth new X.
 * alpha is the migration strength (0.5 = split evenly, the SmoothQuant default).
 */
#ifndef WUBU_SMOOTHQUANT_H
#define WUBU_SMOOTHQUANT_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    int M, K;
    float *W_smooth;   /* [M*K] smoothed weights (W · diag(s)) */
    float *s;          /* [K]  smoothing scales */
    float alpha;
} wubu_smoothquant_t;

/* Build smoothing scales from W and a calibration batch X[nbatch*K].
 * alpha in [0,1]. Returns 0 on success. */
int wubu_smoothquant_init(wubu_smoothquant_t *sq, const float *W, int M, int K,
                           const float *X_calib, int nbatch, float alpha);

/* Smooth a new activation row X_in[K] -> X_out[K] = diag(1/s)·X_in. */
void wubu_smoothquant_activate(const wubu_smoothquant_t *sq, const float *x_in, float *x_out);

/* GEMV with smoothed weights: y[M] = W_smooth @ x_out. */
void wubu_smoothquant_gemv(const wubu_smoothquant_t *sq, const float *x_smoothed, float *y);

void wubu_smoothquant_free(wubu_smoothquant_t *sq);

#endif /* WUBU_SMOOTHQUANT_H */
