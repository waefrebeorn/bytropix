/*
 * wubu_awq.c -- AWQ activation-aware weight quantization (doc B05).
 *
 * Source: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM
 * Compression and Acceleration", MLSys 2024.
 *
 * Core idea: Not all weight channels are equally important. The top ~1%
 * of channels with large activation magnitudes are "salient" — quantizing
 * them naively destroys accuracy. AWQ scales these salient channels UP
 * before quantization (and the corresponding activations DOWN), keeping
 * the matmul result identical while reducing quantization error on the
 * important channels.
 *
 * The key insight: scaling salient channels reduces the relative
 * quantization error for those channels without changing the output.
 *
 * Self-contained C11, no third-party deps.
 */

#include "wubu_awq.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Identify salient channels: top fraction% by activation magnitude. */
void wubu_awq_find_salient(const float *act_magnitudes, int n,
                            float fraction, bool *out_salient) {
    if (!act_magnitudes || !out_salient || n <= 0) return;

    /* Find the threshold: top `fraction` of channels by magnitude */
    /* Simple approach: compute the (1-fraction) percentile */
    float *sorted = (float *)malloc(n * sizeof(float));
    if (!sorted) {
        /* Fallback: mark all as non-salient */
        memset(out_salient, 0, n * sizeof(bool));
        return;
    }
    memcpy(sorted, act_magnitudes, n * sizeof(float));

    /* Simple sort (n is small — this is offline calibration) */
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (sorted[j] > sorted[i]) {
                float tmp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = tmp;
            }
        }
    }

    int n_salient = (int)(n * fraction);
    if (n_salient < 1) n_salient = 1;
    if (n_salient > n) n_salient = n;
    float threshold = sorted[n_salient - 1];

    for (int i = 0; i < n; i++) {
        out_salient[i] = (act_magnitudes[i] >= threshold);
    }
    free(sorted);
}

/* Compute per-channel scaling factor for AWQ.
 * Salient channels get scale > 1 (scale UP weights, DOWN activations).
 * Non-salient channels get scale = 1.
 *
 * The optimal scale for channel i is:
 *   s_i = (avg_act_mag_i / max_weight_i) ^ alpha
 * where alpha controls the trade-off (0 = no scaling, 1 = full AWQ).
 *
 * After scaling: W'[i] = W[i] * s_i, x'[i] = x[i] / s_i
 * Result: W'[i] * x'[i] = W[i] * x[i] (identity), but quant(W * s_i)
 * has less error on salient channels. */
void wubu_awq_compute_scales(const float *act_magnitudes,
                               const float *weight_magnitudes,
                               int n, float alpha,
                               float *out_scales) {
    if (!act_magnitudes || !weight_magnitudes || !out_scales || n <= 0) return;

    /* Find max weight magnitude (for normalization) */
    float max_w = 1e-10f;
    for (int i = 0; i < n; i++) {
        if (weight_magnitudes[i] > max_w)
            max_w = weight_magnitudes[i];
    }

    /* Find max activation magnitude */
    float max_a = 1e-10f;
    for (int i = 0; i < n; i++) {
        if (act_magnitudes[i] > max_a)
            max_a = act_magnitudes[i];
    }

    /* Per-channel scale: s = (act_mag / max_w) ^ alpha */
    for (int i = 0; i < n; i++) {
        float ratio = act_magnitudes[i] / max_w;
        if (ratio < 1e-10f) ratio = 1e-10f;
        out_scales[i] = powf(ratio, alpha);
        /* Clamp to reasonable range */
        if (out_scales[i] < 0.1f) out_scales[i] = 0.1f;
        if (out_scales[i] > 10.0f) out_scales[i] = 10.0f;
        /* Non-salient channels: scale = 1 (no change) */
        if (act_magnitudes[i] < max_a * 0.01f) {
            out_scales[i] = 1.0f;
        }
    }
}

/* Apply AWQ scaling: W_scaled[i] = W[i] * scale[i] */
void wubu_awq_apply_scale_weights(float *W, const float *scales, int n) {
    if (!W || !scales || n <= 0) return;
    for (int i = 0; i < n; i++) {
        W[i] *= scales[i];
    }
}

/* Apply AWQ inverse scaling: x_scaled[i] = x[i] / scale[i]
 * This preserves the matmul (W*s) * (x/s) = W*x */
void wubu_awq_apply_scale_activations(float *x, const float *scales, int n) {
    if (!x || !scales || n <= 0) return;
    for (int i = 0; i < n; i++) {
        x[i] /= scales[i];
    }
}
