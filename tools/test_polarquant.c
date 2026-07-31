/* test_polarquant.c — PolarQuant fractal validation */
#include "wubu_polarquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    printf("=== PolarQuant Fractal Stacking Validation ===\n\n");
    int dims[] = {32, 64, 128};
    int nd = 3;
    int failures = 0;
    for (int di = 0; di < nd; di++) {
        int d = dims[di];
        printf("--- d=%d, depth=%d, bits/coord=2 ---\n", d, WUBU_POLAR_DEPTH);
        wubu_polarquant_t pq;
        if (wubu_polarquant_init(&pq, d, WUBU_POLAR_DEPTH, WUBU_POLAR_R_OUTER, WUBU_POLAR_BITS_PER_COORD) != 0) {
            fprintf(stderr, "pq init FAILED\n"); failures++; continue;
        }
        double bpv = wubu_polarquant_bits_per_vector(&pq, d);
        double cpr = wubu_polarquant_compression_ratio(&pq, d);
        printf("  bits/vec = %.1f  bytes/vec = %.2f  F32 baseline = %d  compress = %.1fx\n",
               bpv, bpv/8.0, d*4, cpr);
        printf("  level radii: [");
        for (int l = 0; l < pq.depth; l++) printf("%.3f ", pq.levels[l].R);
        printf("]\n  level dims:  [");
        for (int l = 0; l < pq.depth; l++) printf("%d ", pq.levels[l].dims);
        printf("]\n");
        /* Validate fractal nesting */
        int nesting_ok = 1;
        for (int l = 1; l < pq.depth; l++) {
            if (pq.levels[l].R >= pq.levels[l-1].R) { nesting_ok = 0; break; }
        }
        printf("  fractal R decreasing: %s\n", nesting_ok ? "PASS" : "FAIL");
        if (!nesting_ok) failures++;
        /* Validate codebook inside ball */
        int inside_ok = 1;
        for (int l = 0; l < pq.depth; l++) {
            float R = pq.levels[l].R;
            for (int k = 0; k < pq.levels[l].codebook_size; k++) {
                float norm = 0.0f;
                const float *p = pq.levels[l].codebook + (size_t)k * pq.levels[l].dims;
                for (int i = 0; i < pq.levels[l].dims; i++) norm += p[i]*p[i];
                if (sqrtf(norm) >= R) { inside_ok = 0; break; }
            }
            if (!inside_ok) break;
        }
        printf("  codebook inside Poincare ball: %s\n", inside_ok ? "PASS" : "FAIL");
        if (!inside_ok) failures++;
        /* 512K bandwidth */
        int n_kv=2, layers=40, ctx=524288;
        double f32_mb = (double)d*4*n_kv*ctx/1e6;
        double pq_mb  = bpv/8.0*n_kv*ctx/1e6;
        printf("  512K KV: F32=%.1f MB  PolarQuant=%.1f MB  reduction=%.0fx\n\n",
               f32_mb, pq_mb, f32_mb/pq_mb);
        wubu_polarquant_free(&pq);
    }
    printf("=== Result: %d validation failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
