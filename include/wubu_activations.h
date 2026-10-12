/* wubu_activations.h — Neural network activation functions
 *
 * Self-contained header: include in any .c file that needs activations.
 * Replaces 11+ independent definitions scattered across the codebase.
 *
 * Usage: #include "wubu_activations.h"
 */

#ifndef WUBU_ACTIVATIONS_H
#define WUBU_ACTIVATIONS_H

#include <math.h>
#include <stdint.h>

/* ---- SiLU (Sigmoid Linear Unit) / Swish ----
 * silu(x) = x * sigmoid(x) = x / (1 + exp(-x)) */

static inline float wubu_act_silu(float x) {
    return x / (1.0f + expf(-x));
}

static inline float wubu_silu_deriv(float x, float silu_x) {
    /* d/dx silu(x) = silu(x) + sigmoid(x) * (1 - silu(x))
     *              = silu(x) * (1 + (1 - silu(x)) / silu_x * x)  [simplified]
     * Simpler: sigmoid(x) = silu(x) / x, so:
     *   d/dx = silu(x)/x + (1 - silu(x)/x) * silu(x)             */
    if (x == 0.0f) return 0.5f;  /* limit as x->0 of sigmoid(x) = 0.5 */
    float sig = silu_x / x;
    return silu_x + sig * (1.0f - silu_x);
}

/* ---- GELU (Gaussian Error Linear Unit) ----
 * gelu(x) = x * Phi(x) where Phi is the standard normal CDF
 * Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */

static inline float wubu_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

/* Fast GELU approximation: x * sigmoid(1.702 * x) */
static inline float wubu_gelu_fast(float x) {
    return x / (1.0f + expf(-1.702f * x));
}

/* Quick GELU: x * sigmoid(x) — used in some CLIP models */
static inline float wubu_quick_gelu(float x) {
    return x / (1.0f + expf(-x));
}

/* ---- ReLU ---- */
static inline float wubu_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

/* ---- LayerNorm ---- */
static inline void wubu_layer_norm(float *out, const float *x, const float *w,
                                    const float *b, int n) {
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    float var = 0.0f;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) {
        out[i] = (x[i] - mean) * inv_std * w[i] + b[i];
    }
}

/* ---- RMSNorm ---- */
/* Returns the RMS scale factor (useful for backward pass) */
static inline float wubu_act_rms_norm(float *out, const float *x, const float *w, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = sqrtf(ss / n + 1e-8f);
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < n; i++) out[i] = x[i] * inv_rms * w[i];
    return inv_rms;
}

/* ---- Softmax ---- */
static inline void wubu_softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv_sum;
}

/* Softmax with temperature: out[i] = exp((in[i] - max) / temp) / sum */
static inline void wubu_softmax_temp(const float *in, float *out, int n, float temp) {
    float max_val = in[0];
    for (int i = 1; i < n; i++) if (in[i] > max_val) max_val = in[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { out[i] = expf((in[i] - max_val) / temp); sum += out[i]; }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) out[i] *= inv_sum;
}

/* 2D softmax over rows */
static inline void wubu_softmax_rows(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) wubu_softmax(x + r * cols, cols);
}

/* ---- Bare-name aliases -------------------------------------------------
 * The model forward/backward files (wubu.c, wubu_backprop.c, wubu_moe*.c,
 * wubu_megakernel.c, wubu_der.c, wubu_generate.c, ...) call these
 * activations by their unprefixed names (silu(x), gelu_fast(x), softmaxf,
 * ...).  These are SELF-CONTAINED scalar implementations (they do NOT call
 * the wubu_silu/wubu_rms_norm names above, which collide with the
 * vectorized buffer versions in wubu_ops.h when both headers are included).
 * (A stray src/wubu_activations.h used to shadow this file via an identical
 * include guard  --  it has been removed; this is the one canonical header.)
 */
static inline float silu(float x)        { return x / (1.0f + expf(-x)); }
static inline float silu_f(float x)      { return x / (1.0f + expf(-x)); }
static inline float sigmoid(float x)     { return 1.0f / (1.0f + expf(-x)); }
static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
static inline float gelu_fast(float x)   { return x / (1.0f + expf(-1.702f * x)); }
static inline float relu(float x)        { return x > 0.0f ? x : 0.0f; }

/* d/dx silu(x).  silu_deriv(x) recomputes; silu_deriv_sg(x, sg) takes the
 * already-computed silu value sg (== silu(x)) to avoid recomputation.
 * d/dx silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))), sigmoid = silu/x. */
static inline float silu_deriv(float x) {
    if (x == 0.0f) return 0.5f;
    float sig = (x / (1.0f + expf(-x))) / x;
    return sig * (1.0f + x * (1.0f - sig));
}
static inline float silu_deriv_sg(float x, float sg) {
    if (x == 0.0f) return 0.5f;
    float sig = sg / x;
    return sig * (1.0f + x * (1.0f - sig));
}

/* bf16 -> f32 (bits shifted into the high half of an IEEE-754 float). */
static inline float bf16_to_f32(uint16_t h) {
    uint32_t u = ((uint32_t)h) << 16;
    float f;
    __builtin_memcpy(&f, &u, sizeof(f));
    return f;
}

/* In-place softmax over n floats (bare name), self-contained. */
static inline void softmaxf(float *a, int n) {
    if (n <= 0) return;
    float mx = a[0];
    for (int i = 1; i < n; i++) if (a[i] > mx) mx = a[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { a[i] = expf(a[i] - mx); sum += a[i]; }
    if (sum > 0.0f) { float inv = 1.0f / sum; for (int i = 0; i < n; i++) a[i] *= inv; }
}

/* Temperature softmax into out[]. */
static inline void softmax_t(const float *in, float *out, int n, float temp) {
    if (n <= 0) return;
    float t = temp > 0.0f ? temp : 1.0f;
    float mx = in[0] / t;
    for (int i = 1; i < n; i++) { float v = in[i] / t; if (v > mx) mx = v; }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { out[i] = expf(in[i] / t - mx); sum += out[i]; }
    if (sum > 0.0f) { float inv = 1.0f / sum; for (int i = 0; i < n; i++) out[i] *= inv; }
}

#endif /* WUBU_ACTIVATIONS_H */
