/*
 * wubu_activations.h  --  scalar activation functions + softmax helpers.
 *
 * The model forward/backward files call these as plain scalar functions
 * (silu(x), sigmoid(x), gelu(x), relu(x), silu_deriv(x), softmaxf, ...).
 * The vectorized wubu_silu/wubu_sigmoid in wubu_ops.c operate on buffers;
 * these are the per-element equivalents used inside element-wise loops.
 * Declared `static inline` so every translation unit that includes this
 * header gets its own copy with no link-time duplicate-symbol clash.
 *
 * NOTE: argmaxf is intentionally NOT here  --  it is a file-local static
 * in wubu_generate.c; do not redefine it.
 */
#ifndef WUBU_ACTIVATIONS_H
#define WUBU_ACTIVATIONS_H

#include <math.h>

/* -- Scalar activations ---------------------------------------------- */

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* silu_f: alias used by the MoE backward pass. */
static inline float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* GELU (tanh approximation) */
static float gelu(float x) {
    const float c = 0.7978845608028654f;  /* sqrt(2/pi) */
    float inner = c * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

/* d/dx silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))) */
static float silu_deriv(float x) {
    float s = sigmoid(x);
    return s * (1.0f + x * (1.0f - s));
}

/* d/dx silu(x) given the precomputed silu value sg (== silu(x)).
 * Since sg = x*sigmoid(x), sigmoid(x) = sg/x (x != 0); avoids recomputing
 * the sigmoid when the forward value is already at hand. */
static float silu_deriv_sg(float x, float sg) {
    float s = (x != 0.0f) ? (sg / x) : sigmoid(x);
    return s * (1.0f + x * (1.0f - s));
}

/* -- Softmax (in-place over a buffer) -------------------------------- */

/* Plain softmax over n floats, in place. */
static void softmaxf(float *a, int n) {
    if (n <= 0) return;
    float mx = a[0];
    for (int i = 1; i < n; i++) if (a[i] > mx) mx = a[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { a[i] = expf(a[i] - mx); sum += a[i]; }
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < n; i++) a[i] *= inv;
    }
}

/* Temperature softmax: divides logits by temperature before normalizing. */
static void softmax_t(float *a, int n, float temperature, float *out) {
    if (n <= 0) return;
    float t = temperature > 0.0f ? temperature : 1.0f;
    float mx = a[0] / t;
    for (int i = 1; i < n; i++) { float v = a[i] / t; if (v > mx) mx = v; }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { out[i] = expf(a[i] / t - mx); sum += out[i]; }
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = 0; i < n; i++) out[i] *= inv;
    }
}

#endif /* WUBU_ACTIVATIONS_H */
