/**
 * test_poincare_router_backward.c — Validate Poincaré router backward.
 */
#include "wubu_moe_hyperbolic.h"
#include "wubu_mobius.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    int B = 1, T = 4, N = B * T, d = D_MODEL;
    float R = R_POINCARE_INPUT, temp = HYPERBOLIC_TEMPERATURE;

    printf("=== Poincaré Router Backward Test ===\n");
    printf("N=%d, D=%d, N_EXPERTS=%d\n", N, d, N_EXPERTS);

    // Create router with random centroids
    poincare_router_t router;
    router.centroids = (float *)malloc((int64_t)N_EXPERTS * d * sizeof(float));
    router.temperature = temp;
    router.loaded = true;

    // Random centroids in ball
    for (int e = 0; e < N_EXPERTS; e++) {
        float *c = router.centroids + e * d;
        float n2 = 0;
        for (int i = 0; i < d; i++) {
            c[i] = ((float)rand()/RAND_MAX - 0.5f) * R * 0.6f;
            n2 += c[i] * c[i];
        }
        float n = sqrtf(n2);
        if (n >= R * 0.9f) { float s = R * 0.85f / n; for (int i=0;i<d;i++) c[i]*=s; }
    }

    // Random input in ball
    float *x = (float *)malloc(N * d * sizeof(float));
    for (int i = 0; i < N * d; i++) x[i] = ((float)rand()/RAND_MAX - 0.5f) * R * 0.5f;

    // Forward
    float *scores = (float *)malloc(N * N_EXPERTS * sizeof(float));
    wubu_poincare_router_forward(x, B, T, &router, scores);
    printf("Forward: score sample [0:2]=%.4e %.4e\n", scores[0], scores[1]);

    // Random d_scores (upstream gradient)
    float *d_scores = (float *)malloc(N * N_EXPERTS * sizeof(float));
    for (int i = 0; i < N * N_EXPERTS; i++) d_scores[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.01f;

    // Backward
    float *d_x = (float *)calloc(N * d, sizeof(float));
    float *d_centroids = (float *)calloc((int64_t)N_EXPERTS * d, sizeof(float));

    wubu_poincare_router_backward(x, B, T, scores, d_scores, &router, d_x, d_centroids);

    // Check gradients
    float max_dx = 0, max_dc = 0;
    for (int i = 0; i < N * d; i++) { float v = fabsf(d_x[i]); if (v > max_dx) max_dx = v; }
    for (int64_t i = 0; i < (int64_t)N_EXPERTS * d; i++) { float v = fabsf(d_centroids[i]); if (v > max_dc) max_dc = v; }

    printf("d_x_max:         %.6e  %s\n", max_dx, max_dx > 0 ? "✓" : "✗");
    printf("d_centroids_max: %.6e  %s\n", max_dc, max_dc > 0 ? "✓" : "✗");

    // Test with NULL pointers
    float *d_x2 = (float *)calloc(N * d, sizeof(float));
    wubu_poincare_router_backward(x, B, T, scores, d_scores, &router, d_x2, NULL);
    float max_dx2 = 0;
    for (int i = 0; i < N * d; i++) { float v = fabsf(d_x2[i]); if (v > max_dx2) max_dx2 = v; }
    printf("d_x (no centroids): %.6e  %s\n", max_dx2, max_dx2 > 0 ? "✓" : "✗");

    int ok = (max_dx > 0 && max_dc > 0 && max_dx2 > 0);
    printf("\n%s\n", ok ? "ALL TESTS PASSED ✓" : "SOME TESTS FAILED ✗");

    free(x); free(scores); free(d_scores);
    free(d_x); free(d_x2); free(d_centroids);
    free(router.centroids);
    return ok ? 0 : 1;
}
