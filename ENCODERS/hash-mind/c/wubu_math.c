/**
 * wubu_math.c — Pure C implementation of WuBu hyperbolic math
 * 
 * Implements Möbius addition and gyration in the Poincaré ball.
 */

#include "wubu_math.h"
#include <stdlib.h>
#include <string.h>

/* ─── Möbius Addition: x ⊕_c y ─── */
/* Formula:
   x ⊕_c y = (1 + 2c⟨x,y⟩ + c‖y‖²)x + (1 - c‖x‖²)y
             ───────────────────────────────────────
             1 + 2c⟨x,y⟩ + c²‖x‖²‖y‖²
*/
void wubu_mobius_add(const float* x, const float* y, float* out, int n, float c) {
    if (c <= 0.0f) {
        for (int i = 0; i < n; i++) out[i] = x[i] + y[i];
        return;
    }
    
    /* Compute dot product ⟨x,y⟩, ‖x‖², ‖y‖² */
    double dot = 0.0, x2 = 0.0, y2 = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)x[i] * y[i];
        x2  += (double)x[i] * x[i];
        y2  += (double)y[i] * y[i];
    }
    
    double c_d = (double)c;
    double denom = 1.0 + 2.0 * c_d * dot + c_d * c_d * x2 * y2 + WUBU_EPS;
    
    double num_factor_x = 1.0 + 2.0 * c_d * dot + c_d * y2;
    double num_factor_y = 1.0 - c_d * x2;
    
    for (int i = 0; i < n; i++) {
        out[i] = (float)((num_factor_x * x[i] + num_factor_y * y[i]) / denom);
    }
    
    wubu_poincare_project(out, n, c);
}

/* ─── Gyration: gyr(u,v)w = (-(u⊕v)) ⊕ (u ⊕ (v ⊕ w)) ─── */
/* Also known as Möbius gyration — measures non-associativity of ⊕ */
void wubu_gyration(const float* u, const float* v, const float* w,
                   float* out, int n, float c) {
    /* Allocate temp buffers */
    float* tmp1 = (float*)malloc(3 * n * sizeof(float));
    if (!tmp1) return;
    float* tmp2 = tmp1 + n;
    float* tmp3 = tmp2 + n;
    
    /* Compute u ⊕ v */
    wubu_mobius_add(u, v, tmp1, n, c);
    
    /* Negate: -(u ⊕ v) */
    for (int i = 0; i < n; i++) tmp1[i] = -tmp1[i];
    
    /* Compute v ⊕ w */
    wubu_mobius_add(v, w, tmp2, n, c);
    
    /* Compute u ⊕ (v ⊕ w) */
    wubu_mobius_add(u, tmp2, tmp3, n, c);
    
    /* Compute -(u⊕v) ⊕ (u ⊕ (v ⊕ w)) */
    wubu_mobius_add(tmp1, tmp3, out, n, c);
    
    free(tmp1);
}

/* ─── Rolling Hash (SimpleHash encoder) ─── */
/* Rabin-Karp style rolling hash: H[i] = Σ_{j=0}^{w-1} val[i+j] * base^{w-1-j} */
/* Forward: H[i+1] = (H[i] - val[i] * base^{w-1}) * base + val[i+w] */
unsigned int wubu_rolling_hash(const int* values, int len, int window_size) {
    if (len < window_size) return 0;
    const unsigned int base = 31;
    const unsigned int modulus = 1000000007;
    
    /* Precompute base^{window_size-1} mod modulus */
    unsigned int base_pow = 1;
    for (int i = 0; i < window_size - 1; i++) {
        base_pow = (base_pow * base) % modulus;
    }
    
    unsigned int hash = 0;
    for (int i = 0; i < window_size; i++) {
        hash = (hash * base + (unsigned int)(values[i] % modulus)) % modulus;
    }
    return hash;
}

/* Compute rolling hashes for entire sequence, stored in output array */
void wubu_rolling_hashes(const int* values, int len, int window_size,
                         unsigned int* output, int* out_len) {
    if (len < window_size) { *out_len = 0; return; }
    
    const unsigned int base = 31;
    const unsigned int modulus = 1000000007;
    
    unsigned int base_pow = 1;
    for (int i = 0; i < window_size - 1; i++) {
        base_pow = (base_pow * base) % modulus;
    }
    
    unsigned int hash = 0;
    for (int i = 0; i < window_size; i++) {
        hash = (hash * base + (unsigned int)(values[i] % modulus)) % modulus;
    }
    output[0] = hash;
    
    int count = 1;
    for (int i = 1; i < len - window_size + 1; i++) {
        unsigned int old_val = (unsigned int)(values[i-1] % modulus);
        unsigned int new_val = (unsigned int)(values[i + window_size - 1] % modulus);
        hash = ((hash + modulus - (old_val * base_pow) % modulus) * base + new_val) % modulus;
        output[count++] = hash;
    }
    *out_len = count;
}
