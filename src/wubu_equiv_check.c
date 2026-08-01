/*
 * wubu_equiv_check.c — Lightweight GEMV equivalence verifier (doc F02). C11.
 */
#include "wubu_equiv_check.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

int wubu_equiv_vectors(const float *a, const float *b, int n,
                        float tol_abs, float *max_abs, float *max_rel) {
    float ma = 0.0f, mr = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        float den = (fabsf(a[i]) > fabsf(b[i])) ? fabsf(a[i]) : fabsf(b[i]);
        float rel = den > 1e-9f ? d / den : d;
        if (d > ma) ma = d;
        if (rel > mr) mr = rel;
    }
    if (max_abs) *max_abs = ma;
    if (max_rel) *max_rel = mr;
    return (ma <= tol_abs && mr <= 1.0f) ? 0 : -1;
}

float wubu_equiv_gemv(wubu_gemv_fn fn, int n, int trials, float tol_rel) {
    float *W = (float *)malloc((size_t)n * n * sizeof(float));
    float *x = (float *)malloc(n * sizeof(float));
    float *out_ref = (float *)malloc(n * sizeof(float));
    float *out_tst = (float *)malloc(n * sizeof(float));
    unsigned seed = 12345u;
    float worst = 0.0f;
    for (int t = 0; t < trials; t++) {
        for (int i = 0; i < n * n; i++) { seed = seed * 1103515245u + 12345u; W[i] = ((float)(seed & 0xFFFF) / 65536.0f) - 0.5f; }
        for (int i = 0; i < n; i++)     { seed = seed * 1103515245u + 12345u; x[i] = ((float)(seed & 0xFFFF) / 65536.0f) - 0.5f; }
        /* naive reference */
        for (int i = 0; i < n; i++) {
            double s = 0.0;
            for (int j = 0; j < n; j++) s += (double)W[(size_t)i * n + j] * (double)x[j];
            out_ref[i] = (float)s;
        }
        fn(out_tst, W, x, n);
        float ma, mr;
        wubu_equiv_vectors(out_ref, out_tst, n, 1e30f, &ma, &mr);
        if (mr > worst) worst = mr;
    }
    if (worst > tol_rel)
        printf("  equiv-check: WORST rel err %.3e > tol %.3e (FAIL)\n", worst, tol_rel);
    free(W); free(x); free(out_ref); free(out_tst);
    return worst;
}
