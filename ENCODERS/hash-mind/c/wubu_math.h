/**
 * wubu_math.h — Pure C port of WuBu JAX hyperbolic math
 * 
 * Core operations from wubu_nesting_impl.py and WuBuMindJAX.py
 * Ported from Python/NumPy/JAX to pure C with no dependencies.
 *
 * Poincaré ball model:
 *   exp_0^c(v) = tanh(√c · ‖v‖) · v / (√c · ‖v‖)
 *   log_0^c(y) = atanh(√c · ‖y‖) · y / (√c · ‖y‖)
 *   x ⊕_c y = Möbius addition
 *   gyr(u,v)w = gyration (Möbius transformation)
 *
 * Curvature c > 0, scale s > 0.
 * All operations use double internally for stability, cast to float for GPU.
 */

#ifndef WUBU_MATH_H
#define WUBU_MATH_H

#include <math.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Constants ─── */
#define WUBU_EPS  1e-7f
#define WUBU_PI   3.14159265358979323846

/* ─── Utilities ─── */
static inline float wubu_clip(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

/* Compute L2 norm of a vector of length n */
static inline float wubu_norm(const float* v, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)v[i] * v[i];
    return (float)sqrt(sum + WUBU_EPS);
}

/* ─── Poincaré Ball Projection ─── */
/* Clip point to stay strictly inside ball of radius 1/sqrt(c) */
static inline void wubu_poincare_project(float* x, int n, float c) {
    if (c <= 0.0f) return;
    float norm = wubu_norm(x, n);
    float max_norm = (1.0f / sqrtf(c)) * (1.0f - WUBU_EPS);
    if (norm > max_norm && norm > 0.0f) {
        float scale = max_norm / norm;
        for (int i = 0; i < n; i++) x[i] *= scale;
    }
}

/* ─── Exponential Map: v ∈ T_0(M) → y ∈ Poincaré ball ─── */
/* exp_0^c(v) = tanh(√c · ‖v‖) · v / (√c · ‖v‖) */
static inline void wubu_exp_map(const float* v, float* y, int n, float c, float scale) {
    if (c <= 0.0f) {
        for (int i = 0; i < n; i++) y[i] = v[i];
        return;
    }
    float norm = wubu_norm(v, n);
    if (norm < WUBU_EPS) {
        for (int i = 0; i < n; i++) y[i] = 0.0f;
        return;
    }
    float sqrt_c = sqrtf(fmaxf(c, WUBU_EPS));
    float scaled_radius = scale * sqrt_c * norm;
    float tanh_term = tanhf(scaled_radius);
    float lambda = tanh_term / (sqrt_c * norm + WUBU_EPS);
    for (int i = 0; i < n; i++) y[i] = lambda * v[i];
    wubu_poincare_project(y, n, c);
}

/* ─── Logarithmic Map: y ∈ Poincaré ball → v ∈ T_0(M) ─── */
/* log_0^c(y) = (1/s)·atanh(√c·‖y‖)·y/(√c·‖y‖) */
static inline void wubu_log_map(const float* y_in, float* v, int n, float c, float scale) {
    if (c <= 0.0f) {
        for (int i = 0; i < n; i++) v[i] = y_in[i];
        return;
    }
    float y_copy[16]; /* max dim we handle inline */
    float* y = (n <= 16) ? y_copy : v;
    for (int i = 0; i < n; i++) y[i] = y_in[i];
    wubu_poincare_project(y, n, c);
    
    float norm = wubu_norm(y, n);
    if (norm < WUBU_EPS) {
        for (int i = 0; i < n; i++) v[i] = 0.0f;
        return;
    }
    float sqrt_c = sqrtf(fmaxf(c, WUBU_EPS));
    float arctanh_input = wubu_clip(sqrt_c * norm, -1.0f + WUBU_EPS, 1.0f - WUBU_EPS);
    float atanh_term = atanhf(arctanh_input) / scale;
    float lambda = atanh_term / (sqrt_c * norm + WUBU_EPS);
    for (int i = 0; i < n; i++) v[i] = lambda * y[i];
}

/* ─── Möbius Addition: x ⊕_c y ─── */
/* Standard formula in Poincaré ball */
void wubu_mobius_add(const float* x, const float* y, float* out, int n, float c);

/* ─── Gyration: gyr(u,v)w = (-(u⊕v)) ⊕ (u ⊕ (v ⊕ w)) ─── */
void wubu_gyration(const float* u, const float* v, const float* w,
                   float* out, int n, float c);

/* ─── Quaternion Hamilton Product ─── */
/* q = w + xi + yj + zk, p = a + bi + cj + dk */
/* q * p = (wa - xb - yc - zd) + (wb + xa + yd - zc)i + ... */
static inline void wubu_hamilton_product(const float q[4], const float p[4], float out[4]) {
    float w1 = q[0], x1 = q[1], y1 = q[2], z1 = q[3];
    float w2 = p[0], x2 = p[1], y2 = p[2], z2 = p[3];
    out[0] = w1*w2 - x1*x2 - y1*y2 - z1*z2;
    out[1] = w1*x2 + x1*w2 + y1*z2 - z1*y2;
    out[2] = w1*y2 - x1*z2 + y1*w2 + z1*x2;
    out[3] = w1*z2 + x1*y2 - y1*x2 + z1*w2;
}

/* ─── Rotation via Quaternion: v' = q * v * q_conj ─── */
static inline void wubu_quaternion_rotate(const float q[4], const float v[3], float out[3]) {
    float p[4] = {0.0f, v[0], v[1], v[2]};
    float q_conj[4] = {q[0], -q[1], -q[2], -q[3]};
    float tmp[4];
    wubu_hamilton_product(q, p, tmp);
    wubu_hamilton_product(tmp, q_conj, out);
    /* out[0] should be ~0, we take imaginary part */
    out[0] = out[1]; out[1] = out[2]; out[2] = out[3];
}

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MATH_H */
