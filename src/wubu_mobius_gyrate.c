/**
 * Optimized Möbius gyration using precomputed dot products.
 *
 * Instead of calling wubu_mobius_add 3 times (which recomputes nx², ny², dxy
 * each call), we compute all needed dot products once and share them.
 */

#include "wubu_mobius.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Internal: mobius_add with precomputed dot products
static void mobius_add_opt(const float *x, const float *y, int d, float R,
                            float nx2, float ny2, float dxy, float *z) {
    float c = 1.0f / (R * R);
    if (nx2 < 1e-30f) { memcpy(z, y, d * sizeof(float)); return; }
    if (ny2 < 1e-30f) { memcpy(z, x, d * sizeof(float)); return; }

    float cny2 = c * ny2, cnx2 = c * nx2;
    float c2nx2ny2 = cnx2 * c * ny2;  // c^2 * nx2 * ny2
    float two_c_dot = 2.0f * c * dxy;
    float alpha = 1.0f + two_c_dot + c2nx2ny2;
    float beta = 1.0f + two_c_dot + cny2;
    float gamma = 1.0f - cnx2;

    if (fabsf(alpha) < 1e-30f) alpha = 1e-30f;
    float inv_alpha = 1.0f / alpha;
    for (int i = 0; i < d; i++)
        z[i] = (beta * x[i] + gamma * y[i]) * inv_alpha;
}

void wubu_mobius_gyrate_opt(const float *x, const float *y, const float *z,
                             int d, float R, float *out) {
    // Precompute dot products and norms
    float nx2 = 0, ny2 = 0, nz2 = 0, dxy = 0, dyz = 0, dxz = 0;
    for (int i = 0; i < d; i++) {
        nx2 += x[i]*x[i]; ny2 += y[i]*y[i]; nz2 += z[i]*z[i];
        dxy += x[i]*y[i]; dyz += y[i]*z[i]; dxz += x[i]*z[i];
    }

    // 1: x⊕y → x_plus_y
    float *x_plus_y = (float *)malloc(d * sizeof(float));
    mobius_add_opt(x, y, d, R, nx2, ny2, dxy, x_plus_y);

    // Norm of x⊕y
    float nx_plus_y2 = 0;
    for (int i = 0; i < d; i++) nx_plus_y2 += x_plus_y[i] * x_plus_y[i];

    // 2: y⊕z → tmp1
    float *tmp1 = (float *)malloc(d * sizeof(float));
    mobius_add_opt(y, z, d, R, ny2, nz2, dyz, tmp1);
    float ntmp1_2 = 0, d_x_tmp1 = 0;
    for (int i = 0; i < d; i++) {
        ntmp1_2 += tmp1[i] * tmp1[i];
        d_x_tmp1 += x[i] * tmp1[i];
    }

    // 3: x ⊕ tmp1 → tmp2
    float *tmp2 = (float *)malloc(d * sizeof(float));
    mobius_add_opt(x, tmp1, d, R, nx2, ntmp1_2, d_x_tmp1, tmp2);
    float ntmp2_2 = 0, d_neg_p_tmp2 = 0;  // = ⟨-x_plus_y, tmp2⟩ = -(x_plus_y · tmp2)
    for (int i = 0; i < d; i++) {
        ntmp2_2 += tmp2[i] * tmp2[i];
        d_neg_p_tmp2 -= x_plus_y[i] * tmp2[i];
    }

    // 4: (-x_plus_y) ⊕ tmp2 → out
    float *neg_x_plus_y = (float *)malloc(d * sizeof(float));
    for (int i = 0; i < d; i++) neg_x_plus_y[i] = -x_plus_y[i];
    mobius_add_opt(neg_x_plus_y, tmp2, d, R, nx_plus_y2, ntmp2_2, d_neg_p_tmp2, out);

    free(neg_x_plus_y);
    free(x_plus_y); free(tmp1); free(tmp2);
}
