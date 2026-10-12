/* wubu_rope_scaling.c — RoPE frequency scaling for context extension (BRAIN #231).
 *
 * The rope forward rotates q/k by pos * inv_freq[i]. To extend context past
 * the trained length, the inverse frequencies are scaled so that positions
 * beyond L_train stay in-distribution. This file computes the scaled
 * inv_freq vector; the measured forward assertions live in test_rope_scaling.
 *
 * C11, self-contained (no GPU, no model deps). YARN delegates to wubu_yarn.
 */
#include "wubu_rope_scaling.h"
#include "wubu_yarn.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

int wubu_rope_scaling_freqs(int half, double base, double s,
                            wubu_rope_scaling_t method,
                            double *out_inv_freq) {
    if (half <= 0 || !out_inv_freq) return -1;
    if (s < 1.0) return -1;
    if (base <= 0.0) return -1;

    switch (method) {
    case WUBU_ROPE_SCALE_NONE:
        for (int i = 0; i < half; i++)
            out_inv_freq[i] = 1.0 / pow(base, (double)(2 * i) / (double)(2 * half));
        return 0;

    case WUBU_ROPE_SCALE_LINEAR:
        /* Divide every frequency by s: the canonical linear (position
         * interpolation) scaling. Exact identity at s==1. */
        for (int i = 0; i < half; i++)
            out_inv_freq[i] = (1.0 / pow(base, (double)(2 * i) / (double)(2 * half))) / s;
        return 0;

    case WUBU_ROPE_SCALE_NTK: {
        /* NTK-aware: scale the rope BASE by s^(d/(d-2)) rather than dividing
         * every frequency. High-frequency dims (small i, weak base dependence)
         * stay ~preserved; low-frequency dims (large i, strong base dependence)
         * compress — avoiding linear scaling's "lost in the middle". d = 2*half. */
        double d = (double)(2 * half);
        double base_ntk = base * pow(s, d / (d - 2.0));
        for (int i = 0; i < half; i++)
            out_inv_freq[i] = 1.0 / pow(base_ntk, (double)(2 * i) / d);
        return 0;
    }

    case WUBU_ROPE_SCALE_YARN: {
        /* YaRN per-dim scale via the existing NTK+ramp machinery. We fold the
         * per-dim scale into the base frequency. beta=16, alpha=1 (YaRN
         * paper defaults for the ramp). */
        double *scale = (double *)malloc(sizeof(double) * (size_t)half);
        double *ramp  = (double *)malloc(sizeof(double) * (size_t)half);
        if (!scale || !ramp) { free(scale); free(ramp); return -1; }
        double L_train = 4096.0;          /* canonical trained length */
        double L_target = L_train * s;
        if (wubu_yarn_scales(2 * half, L_train, L_target, 16.0, scale, ramp) != 0) {
            free(scale); free(ramp); return -1;
        }
        for (int i = 0; i < half; i++)
            out_inv_freq[i] = (1.0 / pow(base, (double)(2 * i) / (double)(2 * half))) * scale[i];
        free(scale); free(ramp);
        return 0;
    }
    }
    return -1;
}

const char *wubu_rope_scaling_name(wubu_rope_scaling_t method) {
    switch (method) {
    case WUBU_ROPE_SCALE_NONE:   return "none";
    case WUBU_ROPE_SCALE_LINEAR: return "linear";
    case WUBU_ROPE_SCALE_NTK:    return "ntk";
    case WUBU_ROPE_SCALE_YARN:   return "yarn";
    }
    return "unknown";
}
