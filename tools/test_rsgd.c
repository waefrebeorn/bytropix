/**
 * test_rsgd.c — Verify RSGD optimizer produces valid Poincaré ball output
 */
#include "rsgd.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    const int n = 1000, d = 128;
    float R = 0.956f;
    
    // Generate random vectors already in Poincaré ball
    float *w = (float *)malloc(n * d * sizeof(float));
    for (int i = 0; i < n * d; i++)
        w[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f; // |v| < 0.5
    
    // Random gradients
    float *dw = (float *)malloc(n * d * sizeof(float));
    for (int i = 0; i < n * d; i++)
        dw[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    
    // Check initial norms are ≤ R
    double max_init_norm = 0;
    for (int v = 0; v < n; v++) {
        double n2 = 0;
        for (int i = 0; i < d; i++) n2 += w[v*d+i] * w[v*d+i];
        double norm = sqrt(n2);
        if (norm > max_init_norm) max_init_norm = norm;
    }
    printf("Initial max norm: %.4f (R=%.4f)\n", max_init_norm, R);
    
    // Apply RSGD step
    rsgd_step(w, dw, n, d, 0.01f, R, 1.0f);
    
    // Check final norms are ≤ R
    double max_final_norm = 0;
    int nan_count = 0, inf_count = 0;
    for (int v = 0; v < n; v++) {
        double n2 = 0;
        for (int i = 0; i < d; i++) {
            float val = w[v*d+i];
            if (isnan(val)) { nan_count++; break; }
            if (isinf(val)) { inf_count++; break; }
            n2 += val * val;
        }
        double norm = sqrt(n2);
        if (norm > max_final_norm) {
            if (!isnan(n2) && !isinf(n2))
                max_final_norm = norm;
        }
    }
    printf("Final max norm: %.4f (R=%.4f)\n", max_final_norm, R);
    printf("NaN: %d, Inf: %d\n", nan_count, inf_count);
    
    if (max_final_norm <= R * 1.001 && nan_count == 0 && inf_count == 0)
        printf("PASS: RSGD produces valid Poincaré ball output\n");
    else
        printf("FAIL: RSGD produced out-of-ball or NaN output\n");
    
    return 0;
}
