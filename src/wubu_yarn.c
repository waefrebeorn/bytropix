/*
 * wubu_yarn.c — YaRN NTK-aware RoPE extrapolation (Round-3 #241/#242).
 * C11, self-contained. Extends a model's trained context to longer lengths by
 * applying an NTK-aware scaling + dimensional ramp: low-frequency dims use more
 * extrapolation, high-frequency dims use less (preserving local detail). Gives
 * Qwen3.6 its 262K -> 1M context extension.
 */
#include "wubu_yarn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* Compute per-dim scaling vector for YaRN given trained dim `d`, trained ctx
 * `L_train`, target ctx `L_target`, ramp `beta` (typ 16) and alpha. Writes
 * `scale` (d/2 entries) and `ramp` (d/2). Returns 0 on ok, -1 on bad args. */
int wubu_yarn_scales(int d, double L_train, double L_target,
                     double beta, double *scale, double *ramp) {
    if (d <= 0 || L_train <= 0 || L_target <= L_train) return -1;
    int half = d / 2;
    double alpha = (L_target / L_train) - 1.0;       /* extrapolation factor */
    /* NTK-aware: high-freq dims (i small, ramp~0) keep scale 1; low-freq dims
     * (i large, ramp~1) are extrapolated up to (L_target/L_train)^alpha. */
    for (int i = 0; i < half; i++) {
        double frac = (double)i / (half - 1);
        double r = 0.5 * (1.0 + tanh(beta * (frac - 0.5) / 0.5)); /* 0..1 ramp */
        ramp[i] = r;
        scale[i] = pow(L_target / L_train, r * alpha);  /* high-freq ~1, low-freq up */
    }
    return 0;
}

/* Apply scaling to a rotary angle pair (theta). theta already = pos * freq. */
void wubu_yarn_apply(double theta, const double *scale, int half, int dim_idx,
                     double *theta_out) {
    int s = dim_idx / 2;
    if (s < 0) s = 0;
    if (s >= half) s = half - 1;
    *theta_out = theta * scale[s];
}
