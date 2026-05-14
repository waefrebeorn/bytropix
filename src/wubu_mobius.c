#include "wubu_mobius.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================
// Möbius addition
// ============================================================
void wubu_mobius_add(const float *x, const float *y, int d, float R, float *z) {
    // The numerically stable formula from Ganea et al. (2018):
    // Let c = 1/R^2 (curvature).
    // Then:
    //   x ⊕ y = (1 + 2c⟨x,y⟩ + c||y||²) x + (1 - c||x||²) y
    //            ────────────────────────────────────────────────
    //              1 + 2c⟨x,y⟩ + c²||x||²||y||²
    //
    // But we work with R directly, so c = 1/R^2.
    
    float c = 1.0f / (R * R);
    
    // Compute dot product and squared norms
    float dot_xy = 0.0f;
    float nx2 = 0.0f, ny2 = 0.0f;
    for (int i = 0; i < d; i++) {
        dot_xy += x[i] * y[i];
        nx2 += x[i] * x[i];
        ny2 += y[i] * y[i];
    }
    
    // Check if x is at origin (zero norm)
    if (nx2 < 1e-30f) {
        // 0 ⊕ y = y
        memcpy(z, y, d * sizeof(float));
        return;
    }
    if (ny2 < 1e-30f) {
        // x ⊕ 0 = x
        memcpy(z, x, d * sizeof(float));
        return;
    }
    
    // Compute numerator coefficients
    float cny2 = c * ny2;
    float cnx2 = c * nx2;
    float c2nx2ny2 = c * cnx2 * ny2;
    float two_c_dot = 2.0f * c * dot_xy;
    
    float denom = 1.0f + two_c_dot + c2nx2ny2;
    float num1 = 1.0f + two_c_dot + cny2;
    float num2 = 1.0f - cnx2;
    
    // Clamp denominator to avoid division by zero
    // In theory denom > 0 for all valid ball points, but with float32
    // boundary conditions we might get near-zero.
    if (fabsf(denom) < 1e-30f) {
        // Fallback: just return y (boundary case — shouldn't happen with valid inputs)
        memcpy(z, y, d * sizeof(float));
        return;
    }
    
    float inv_denom = 1.0f / denom;
    
    // z = (num1 * x + num2 * y) / denom
    for (int i = 0; i < d; i++) {
        z[i] = (num1 * x[i] + num2 * y[i]) * inv_denom;
    }
}

// ============================================================
// Möbius scalar multiplication
// ============================================================
void wubu_mobius_scalar_mul(float r, const float *x, int d, float R, float *z) {
    // r ⊗ x = tanh(r * artanh(||x||/R)) * x / ||x|| * R
    // = exp_map(r * log_map(x))
    
    float nx = 0.0f;
    for (int i = 0; i < d; i++) nx += x[i] * x[i];
    nx = sqrtf(nx);
    
    // If x is at origin, result is origin (regardless of r)
    if (nx < 1e-30f) {
        memset(z, 0, d * sizeof(float));
        return;
    }
    
    // Compute scaling factor
    float ratio = nx / R;
    // Clamp to avoid domain error in artanh (must be < 1)
    if (ratio >= 0.9999f) ratio = 0.9999f;
    
    float artanh_r = 0.5f * logf((1.0f + ratio) / (1.0f - ratio));
    float scale = tanhf(r * artanh_r);
    float factor = scale * R / nx;
    
    for (int i = 0; i < d; i++) {
        z[i] = x[i] * factor;
    }
}

// ============================================================
// Poincaré geodesic distance
// ============================================================
float wubu_poincare_dist(const float *x, const float *y, int d, float R) {
    // d(x,y) = R * artanh(||(-x) ⊕ y|| / R)
    float *neg_x = (float *)malloc(d * sizeof(float));
    for (int i = 0; i < d; i++) neg_x[i] = -x[i];
    
    float *diff = (float *)malloc(d * sizeof(float));
    wubu_mobius_add(neg_x, y, d, R, diff);
    
    float ndiff = 0.0f;
    for (int i = 0; i < d; i++) ndiff += diff[i] * diff[i];
    ndiff = sqrtf(ndiff);
    
    // Clamp for artanh domain
    float ratio = ndiff / R;
    if (ratio >= 0.9999f) ratio = 0.9999f;
    if (ratio < 0.0f) ratio = 0.0f;
    
    float dist = R * 0.5f * logf((1.0f + ratio) / (1.0f - ratio));
    
    free(neg_x);
    free(diff);
    return dist;
}

// ============================================================
// Tangent-space linear combination (Poincaré I)
// ============================================================
void wubu_poincare_linear_comb(const float **xi, const float *wi, int n, int d, float R, float *z) {
    // z = exp_map(sum_i w_i * log_map(x_i))
    
    // Allocate tangent-space buffer
    float *tangent = (float *)calloc(d, sizeof(float));
    if (!tangent) return;
    
    float *log_xi = (float *)malloc(d * sizeof(float));
    if (!log_xi) { free(tangent); return; }
    
    // Sum w_i * log_map(x_i) in tangent space
    for (int i = 0; i < n; i++) {
        // log_map(x) = R * artanh(||x||/R) * x/||x||
        float nx = 0.0f;
        for (int j = 0; j < d; j++) nx += xi[i][j] * xi[i][j];
        nx = sqrtf(nx);
        
        if (nx < 1e-30f) continue;  // skip origin points
        
        float ratio = nx / R;
        if (ratio >= 0.9999f) ratio = 0.9999f;
        float artanh_r = 0.5f * logf((1.0f + ratio) / (1.0f - ratio));
        float scale = R * artanh_r / nx;
        
        for (int j = 0; j < d; j++) {
            tangent[j] += wi[i] * xi[i][j] * scale;
        }
    }
    
    // exp_map(tangent) = tanh(||t||/R) * t/||t|| * R
    float nt = 0.0f;
    for (int j = 0; j < d; j++) nt += tangent[j] * tangent[j];
    nt = sqrtf(nt);
    
    if (nt < 1e-30f) {
        memset(z, 0, d * sizeof(float));
    } else {
        float ratio = nt / R;
        if (ratio >= 0.9999f) ratio = 0.9999f;
        float tanh_r = tanhf(ratio);
        float factor = tanh_r * R / nt;
        for (int j = 0; j < d; j++) {
            z[j] = tangent[j] * factor;
        }
    }
    
    free(tangent);
    free(log_xi);
}

// ============================================================
// Möbius gyration operator (stub for now)
// ============================================================
void wubu_mobius_gyrate(const float *x, const float *y, const float *z, int d, float R, float *out) {
    // gyr[x,y]z = (-(x ⊕ y)) ⊕ (x ⊕ (y ⊕ z))
    // Also equals: z + 2*α*⟨x,z⟩*x + 2*β*⟨y,z⟩*y + 2*γ*⟨x,z⟩*y + 2*δ*⟨y,z⟩*x
    // where α, β, γ, δ are scalar functions of x,y.
    //
    // For now, fall back to full Möbius composition.
    // This is expensive (3 Möbius adds) but correct.
    // Optimize with the closed-form gyration formula in the future.
    
    float *tmp1 = (float *)malloc(d * sizeof(float));
    float *tmp2 = (float *)malloc(d * sizeof(float));
    float *x_plus_y = (float *)malloc(d * sizeof(float));
    
    wubu_mobius_add(x, y, d, R, x_plus_y);
    // -(x ⊕ y)
    for (int i = 0; i < d; i++) tmp1[i] = -x_plus_y[i];
    
    // y ⊕ z
    wubu_mobius_add(y, z, d, R, tmp2);
    // x ⊕ (y ⊕ z)
    wubu_mobius_add(x, tmp2, d, R, tmp2);
    
    // gyr[x,y]z = (-(x⊕y)) ⊕ (x⊕(y⊕z))
    wubu_mobius_add(tmp1, tmp2, d, R, out);
    
    free(tmp1);
    free(tmp2);
    free(x_plus_y);
}
