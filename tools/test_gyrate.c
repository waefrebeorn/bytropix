/**
 * test_gyrate.c — Validate optimized gyration against reference 3-add version.
 */
#include "wubu_mobius.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Reference: original 3-add implementation from wubu_mobius.c
static void gyr_ref(const float *x, const float *y, const float *z, int d, float R, float *out) {
    float *tmp1 = (float *)malloc(d * sizeof(float));
    float *tmp2 = (float *)malloc(d * sizeof(float));
    float *x_plus_y = (float *)malloc(d * sizeof(float));
    wubu_mobius_add(x, y, d, R, x_plus_y);
    for (int i = 0; i < d; i++) tmp1[i] = -x_plus_y[i];
    wubu_mobius_add(y, z, d, R, tmp2);
    wubu_mobius_add(x, tmp2, d, R, tmp2);
    wubu_mobius_add(tmp1, tmp2, d, R, out);
    free(tmp1); free(tmp2); free(x_plus_y);
}

extern void wubu_mobius_gyrate_opt(const float *x, const float *y, const float *z,
                                    int d, float R, float *out);

int main() {
    int d = 16;
    float R = 2.0f;

    float x[16], y[16], z[16];
    float ref[16], opt[16];

    printf("=== Gyration Optimization Test ===\n");
    printf("d=%d, R=%.1f\n\n", d, R);

    int num_tests = 10;
    float max_err = 0;

    for (int t = 0; t < num_tests; t++) {
        // Random vectors inside ball
        for (int i = 0; i < d; i++) {
            x[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.8f;
            y[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.8f;
            z[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.8f;
        }
        // Scale to ensure within ball (||v|| < R)
        { float *vecs[] = {x,y,z};
          for (int vi = 0; vi < 3; vi++) {
            float *v = vecs[vi];
            float n2 = 0; for (int i=0;i<d;i++) n2 += v[i]*v[i];
            float n = sqrtf(n2);
            if (n >= R * 0.9f) { float s = R * 0.85f / n; for (int i=0;i<d;i++) v[i] *= s; }
          }
        }

        gyr_ref(x, y, z, d, R, ref);
        wubu_mobius_gyrate_opt(x, y, z, d, R, opt);

        float err = 0;
        for (int i = 0; i < d; i++) {
            float e = fabsf(ref[i] - opt[i]);
            if (e > err) err = e;
        }
        printf("Test %d: max_err=%.6e  %s\n", t+1, err, err < 1e-5 ? "✓" : "✗");
        if (err > max_err) max_err = err;

        // Validate output in ball
        float n2 = 0; for (int i=0;i<d;i++) n2 += opt[i]*opt[i];
        if (sqrtf(n2) >= R) printf("  WARNING: output out of ball (||gyr||=%.4f >= R=%.1f)\n", sqrtf(n2), R);
    }

    printf("\nMax error vs reference: %.6e  %s\n", max_err, max_err < 1e-5 ? "PASS ✓" : "FAIL ✗");
    return max_err < 1e-5 ? 0 : 1;
}
