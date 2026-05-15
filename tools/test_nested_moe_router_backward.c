/**
 * test_nested_moe_router_backward.c — Validate Nested MoE 2-level backward pass.
 *
 * Tests:
 *   1. Forward runs, backward runs without crash
 *   2. Gradients w.r.t. input, coarse centroids, fine centroids are non-zero
 *   3. Gradients are finite (no NaN/Inf)
 *   4. NULL pointer guards (individual gradient buffers can be NULL)
 *
 * Build: gcc -O2 -I include -o test_nested_moe_router_backward \
 *        tools/test_nested_moe_router_backward.c \
 *        src/wubu_moe_hyperbolic.o src/wubu_mobius.o src/wubu_moe_hyperbolic_backward.o \
 *        -lm -fopenmp
 */

#include "wubu_moe_hyperbolic.h"
#include "wubu_moe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Test dimensions
#define TEST_B 1
#define TEST_T 4
#define TEST_N (TEST_B * TEST_T)  // 4 tokens

static int tests_passed = 0;
static int tests_total = 0;

static void check_true(const char *name, int cond) {
    tests_total++;
    if (cond) {
        printf("  PASS: %s\n", name);
        tests_passed++;
    } else {
        printf("  FAIL: %s\n", name);
    }
}

__attribute__((unused)) static void check_float(const char *name, float got, float expected, float tol) {
    tests_total++;
    if (fabsf(got - expected) <= tol) {
        printf("  PASS: %s = %.6e\n", name, got);
        tests_passed++;
    } else {
        printf("  FAIL: %s = %.6e (expected %.6e, tol %.6e)\n", name, got, expected, tol);
    }
}

// ============================================================
// Test 1: Basic forward + backward — gradients non-zero and finite
// ============================================================
static void test_basic_backward(void) {
    printf("\n=== Test 1: Basic Nested MoE Backward ===\n");

    int d = D_MODEL;

    // Create router with random centroids
    float *coarse = (float *)malloc(N_HYPERBOLIC_GROUPS * d * sizeof(float));
    float *fine = (float *)malloc(N_EXPERTS * d * sizeof(float));
    wubu_nested_moe_router_init_random(coarse, fine, 42);

    nested_moe_router_t router;
    router.coarse_centroids = coarse;
    router.fine_centroids = fine;
    router.temperature = HYPERBOLIC_TEMPERATURE;
    router.loaded = true;

    // Random input
    float *x = (float *)malloc(TEST_N * d * sizeof(float));
    for (int i = 0; i < TEST_N * d; i++)
        x[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;

    // Forward
    int *indices = (int *)malloc(TEST_N * N_ACTIVE_EXPTS * sizeof(int));
    float *weights = (float *)malloc(TEST_N * N_ACTIVE_EXPTS * sizeof(float));
    wubu_nested_moe_router_forward(x, TEST_B, TEST_T, &router, indices, weights);

    // Print forward results
    printf("  Forward weight sample: ");
    for (int k = 0; k < N_ACTIVE_EXPTS; k++)
        printf("%.4f ", weights[k]);
    printf("\n");
    printf("  Forward index sample: ");
    for (int k = 0; k < N_ACTIVE_EXPTS; k++)
        printf("%d ", indices[k]);
    printf("\n");

    // Random upstream gradients
    float *d_weights = (float *)malloc(TEST_N * N_ACTIVE_EXPTS * sizeof(float));
    for (int i = 0; i < TEST_N * N_ACTIVE_EXPTS; i++)
        d_weights[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;

    // Backward: all gradients
    float *d_x = (float *)calloc(TEST_N * d, sizeof(float));
    float *d_coarse = (float *)calloc(N_HYPERBOLIC_GROUPS * d, sizeof(float));
    float *d_fine = (float *)calloc(N_EXPERTS * d, sizeof(float));

    wubu_nested_moe_router_backward(x, TEST_B, TEST_T,
                                     indices, weights, d_weights,
                                     &router,
                                     d_x, d_coarse, d_fine);

    // Check gradients are non-zero
    float max_dx = 0, max_dc = 0, max_df = 0;
    for (int i = 0; i < TEST_N * d; i++) { float v = fabsf(d_x[i]); if (v > max_dx) max_dx = v; }
    for (int i = 0; i < N_HYPERBOLIC_GROUPS * d; i++) { float v = fabsf(d_coarse[i]); if (v > max_dc) max_dc = v; }
    for (int64_t i = 0; i < (int64_t)N_EXPERTS * d; i++) { float v = fabsf(d_fine[i]); if (v > max_df) max_df = v; }

    printf("  max |d_x|:        %.6e\n", max_dx);
    printf("  max |d_coarse|:   %.6e\n", max_dc);
    printf("  max |d_fine|:     %.6e\n", max_df);

    check_true("d_x has non-zero gradients", max_dx > 0);
    check_true("d_coarse has non-zero gradients", max_dc > 0);
    check_true("d_fine has non-zero gradients", max_df > 0);

    // Check for NaN/Inf
    int nan_or_inf = 0;
    for (int i = 0; i < TEST_N * d; i++) {
        if (isnan(d_x[i]) || isinf(d_x[i])) { nan_or_inf = 1; break; }
    }
    for (int i = 0; i < N_HYPERBOLIC_GROUPS * d; i++) {
        if (isnan(d_coarse[i]) || isinf(d_coarse[i])) { nan_or_inf = 1; break; }
    }
    for (int64_t i = 0; i < (int64_t)N_EXPERTS * d; i++) {
        if (isnan(d_fine[i]) || isinf(d_fine[i])) { nan_or_inf = 1; break; }
    }
    check_true("no NaN or Inf in gradients", !nan_or_inf);

    free(x); free(indices); free(weights); free(d_weights);
    free(d_x); free(d_coarse); free(d_fine);
    free(coarse); free(fine);
}

// ============================================================
// Test 2: NULL pointer guards
// ============================================================
static void test_null_guards(void) {
    printf("\n=== Test 2: NULL Pointers in Backward ===\n");

    int d = D_MODEL;

    float *coarse = (float *)malloc(N_HYPERBOLIC_GROUPS * d * sizeof(float));
    float *fine = (float *)malloc(N_EXPERTS * d * sizeof(float));
    wubu_nested_moe_router_init_random(coarse, fine, 99);

    nested_moe_router_t router;
    router.coarse_centroids = coarse;
    router.fine_centroids = fine;
    router.temperature = HYPERBOLIC_TEMPERATURE;
    router.loaded = true;

    float *x = (float *)malloc(TEST_N * d * sizeof(float));
    for (int i = 0; i < TEST_N * d; i++)
        x[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;

    int *indices = (int *)malloc(TEST_N * N_ACTIVE_EXPTS * sizeof(int));
    float *weights = (float *)malloc(TEST_N * N_ACTIVE_EXPTS * sizeof(float));
    wubu_nested_moe_router_forward(x, TEST_B, TEST_T, &router, indices, weights);

    float *d_weights = (float *)malloc(TEST_N * N_ACTIVE_EXPTS * sizeof(float));
    for (int i = 0; i < TEST_N * N_ACTIVE_EXPTS; i++)
        d_weights[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;

    // Test A: NULL d_x — allocate separate buffers
    float *d_coarse_a = (float *)calloc(N_HYPERBOLIC_GROUPS * d, sizeof(float));
    float *d_fine_a = (float *)calloc(N_EXPERTS * d, sizeof(float));
    wubu_nested_moe_router_backward(x, TEST_B, TEST_T,
                                     indices, weights, d_weights,
                                     &router,
                                     NULL, d_coarse_a, d_fine_a);
    float max_c = 0;
    for (int i = 0; i < N_HYPERBOLIC_GROUPS * d; i++) { float v = fabsf(d_coarse_a[i]); if (v > max_c) max_c = v; }
    check_true("NULL d_x: coarse gradients non-zero", max_c > 0);
    float max_f = 0;
    for (int64_t i = 0; i < (int64_t)N_EXPERTS * d; i++) { float v = fabsf(d_fine_a[i]); if (v > max_f) max_f = v; }
    check_true("NULL d_x: fine gradients non-zero", max_f > 0);
    printf("  (d_coarse_max=%.6e, d_fine_max=%.6e)\n", max_c, max_f);
    free(d_coarse_a); free(d_fine_a);

    // Test B: NULL d_coarse_centroids
    float *d_x_b = (float *)calloc(TEST_N * d, sizeof(float));
    float *d_fine_b = (float *)calloc(N_EXPERTS * d, sizeof(float));
    wubu_nested_moe_router_backward(x, TEST_B, TEST_T,
                                     indices, weights, d_weights,
                                     &router,
                                     d_x_b, NULL, d_fine_b);
    max_f = 0;
    for (int64_t i = 0; i < (int64_t)N_EXPERTS * d; i++) { float v = fabsf(d_fine_b[i]); if (v > max_f) max_f = v; }
    check_true("NULL coarse: fine gradients non-zero", max_f > 0);
    float max_x = 0;
    for (int i = 0; i < TEST_N * d; i++) { float v = fabsf(d_x_b[i]); if (v > max_x) max_x = v; }
    check_true("NULL coarse: d_x gradients non-zero", max_x > 0);
    printf("  (d_fine_max=%.6e, d_x_max=%.6e)\n", max_f, max_x);
    free(d_x_b); free(d_fine_b);

    // Test C: NULL d_fine_centroids
    float *d_x_c = (float *)calloc(TEST_N * d, sizeof(float));
    float *d_coarse_c = (float *)calloc(N_HYPERBOLIC_GROUPS * d, sizeof(float));
    wubu_nested_moe_router_backward(x, TEST_B, TEST_T,
                                     indices, weights, d_weights,
                                     &router,
                                     d_x_c, d_coarse_c, NULL);
    max_c = 0;
    for (int i = 0; i < N_HYPERBOLIC_GROUPS * d; i++) { float v = fabsf(d_coarse_c[i]); if (v > max_c) max_c = v; }
    check_true("NULL fine: coarse gradients non-zero", max_c > 0);
    max_x = 0;
    for (int i = 0; i < TEST_N * d; i++) { float v = fabsf(d_x_c[i]); if (v > max_x) max_x = v; }
    check_true("NULL fine: d_x gradients non-zero", max_x > 0);
    printf("  (d_coarse_max=%.6e, d_x_max=%.6e)\n", max_c, max_x);
    free(d_x_c); free(d_coarse_c);

    // Test D: NULL d_fine + d_coarse
    float *d_x_d = (float *)calloc(TEST_N * d, sizeof(float));
    wubu_nested_moe_router_backward(x, TEST_B, TEST_T,
                                     indices, weights, d_weights,
                                     &router,
                                     d_x_d, NULL, NULL);
    max_x = 0;
    for (int i = 0; i < TEST_N * d; i++) { float v = fabsf(d_x_d[i]); if (v > max_x) max_x = v; }
    check_true("only d_x: gradients non-zero", max_x > 0);
    printf("  (d_x_max=%.6e)\n", max_x);
    free(d_x_d);

    // Test E: NULL router (should silently return)
    float *d_x_e = (float *)calloc(TEST_N * d, sizeof(float));
    float *d_coarse_e = (float *)calloc(N_HYPERBOLIC_GROUPS * d, sizeof(float));
    float *d_fine_e = (float *)calloc(N_EXPERTS * d, sizeof(float));
    router.loaded = false;
    wubu_nested_moe_router_backward(x, TEST_B, TEST_T,
                                     indices, weights, d_weights,
                                     &router,
                                     d_x_e, d_coarse_e, d_fine_e);
    max_x = 0;
    for (int i = 0; i < TEST_N * d; i++) { float v = fabsf(d_x_e[i]); if (v > max_x) max_x = v; }
    check_true("unloaded router: d_x all zero", max_x == 0);
    free(d_x_e); free(d_coarse_e); free(d_fine_e);
    // Re-enable router
    router.loaded = true;

    free(x); free(indices); free(weights); free(d_weights);
    free(coarse); free(fine);
}

// ============================================================
// Test 3: Weight sum sanity — verify gradient sign for uniform weights
// ============================================================
static void test_uniform_weight_gradient(void) {
    printf("\n=== Test 3: Uniform Weight Gradient ===\n");

    int d = D_MODEL;

    float *coarse = (float *)malloc(N_HYPERBOLIC_GROUPS * d * sizeof(float));
    float *fine = (float *)malloc(N_EXPERTS * d * sizeof(float));
    wubu_nested_moe_router_init_random(coarse, fine, 77);

    nested_moe_router_t router;
    router.coarse_centroids = coarse;
    router.fine_centroids = fine;
    router.temperature = HYPERBOLIC_TEMPERATURE;
    router.loaded = true;

    // Single token
    float *x = (float *)malloc(1 * d * sizeof(float));
    for (int i = 0; i < d; i++)
        x[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;

    int *indices = (int *)malloc(1 * N_ACTIVE_EXPTS * sizeof(int));
    float *weights = (float *)malloc(1 * N_ACTIVE_EXPTS * sizeof(float));
    wubu_nested_moe_router_forward(x, 1, 1, &router, indices, weights);

    // Upstream gradient = non-uniform values (more realistic than all-ones)
    float *d_weights = (float *)malloc(1 * N_ACTIVE_EXPTS * sizeof(float));
    for (int k = 0; k < N_ACTIVE_EXPTS; k++) d_weights[k] = (float)(k + 1) * 0.01f;  // 0.01, 0.02, ..., 0.08

    float *d_x = (float *)calloc(1 * d, sizeof(float));
    float *d_coarse = (float *)calloc(N_HYPERBOLIC_GROUPS * d, sizeof(float));
    float *d_fine = (float *)calloc(N_EXPERTS * d, sizeof(float));

    wubu_nested_moe_router_backward(x, 1, 1,
                                     indices, weights, d_weights,
                                     &router,
                                     d_x, d_coarse, d_fine);

    // Check gradients are finite and non-zero
    float max_dx = 0;
    for (int i = 0; i < d; i++) { float v = fabsf(d_x[i]); if (v > max_dx) max_dx = v; }
    printf("  max |d_x|: %.6e (should be > 0)\n", max_dx);
    check_true("d_x non-zero", max_dx > 0);

    // Check no NaN
    int bad = 0;
    for (int i = 0; i < d; i++) if (isnan(d_x[i]) || isinf(d_x[i])) { bad = 1; break; }
    for (int i = 0; i < N_HYPERBOLIC_GROUPS * d; i++) if (isnan(d_coarse[i]) || isinf(d_coarse[i])) { bad = 1; break; }
    for (int64_t i = 0; i < (int64_t)N_EXPERTS * d; i++) if (isnan(d_fine[i]) || isinf(d_fine[i])) { bad = 1; break; }
    check_true("no NaN/Inf", !bad);

    free(x); free(indices); free(weights); free(d_weights);
    free(d_x); free(d_coarse); free(d_fine);
    free(coarse); free(fine);
}

// ============================================================
// Test 4: Deterministic — same input gives same gradient
// ============================================================
static void test_deterministic(void) {
    printf("\n=== Test 4: Deterministic Gradients ===\n");

    int d = D_MODEL;

    float *coarse = (float *)malloc(N_HYPERBOLIC_GROUPS * d * sizeof(float));
    float *fine = (float *)malloc(N_EXPERTS * d * sizeof(float));
    wubu_nested_moe_router_init_random(coarse, fine, 123);

    nested_moe_router_t router;
    router.coarse_centroids = coarse;
    router.fine_centroids = fine;
    router.temperature = HYPERBOLIC_TEMPERATURE;
    router.loaded = true;

    float *x = (float *)malloc(TEST_N * d * sizeof(float));
    for (int i = 0; i < TEST_N * d; i++)
        x[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;

    int *indices = (int *)malloc(TEST_N * N_ACTIVE_EXPTS * sizeof(int));
    float *weights = (float *)malloc(TEST_N * N_ACTIVE_EXPTS * sizeof(float));
    wubu_nested_moe_router_forward(x, TEST_B, TEST_T, &router, indices, weights);

    float *d_weights = (float *)malloc(TEST_N * N_ACTIVE_EXPTS * sizeof(float));
    for (int i = 0; i < TEST_N * N_ACTIVE_EXPTS; i++)
        d_weights[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;

    // Run backward twice
    float *d_x1 = (float *)calloc(TEST_N * d, sizeof(float));
    float *d_c1 = (float *)calloc(N_HYPERBOLIC_GROUPS * d, sizeof(float));
    float *d_f1 = (float *)calloc(N_EXPERTS * d, sizeof(float));
    wubu_nested_moe_router_backward(x, TEST_B, TEST_T,
                                     indices, weights, d_weights,
                                     &router,
                                     d_x1, d_c1, d_f1);

    float *d_x2 = (float *)calloc(TEST_N * d, sizeof(float));
    float *d_c2 = (float *)calloc(N_HYPERBOLIC_GROUPS * d, sizeof(float));
    float *d_f2 = (float *)calloc(N_EXPERTS * d, sizeof(float));
    wubu_nested_moe_router_backward(x, TEST_B, TEST_T,
                                     indices, weights, d_weights,
                                     &router,
                                     d_x2, d_c2, d_f2);

    // Compare
    int match = 1;
    for (int i = 0; i < TEST_N * d; i++)
        if (fabsf(d_x1[i] - d_x2[i]) > 1e-6f) { match = 0; break; }
    for (int i = 0; i < N_HYPERBOLIC_GROUPS * d; i++)
        if (fabsf(d_c1[i] - d_c2[i]) > 1e-6f) { match = 0; break; }
    for (int64_t i = 0; i < (int64_t)N_EXPERTS * d; i++)
        if (fabsf(d_f1[i] - d_f2[i]) > 1e-6f) { match = 0; break; }

    check_true("deterministic gradients (run 1 == run 2)", match);

    free(x); free(indices); free(weights); free(d_weights);
    free(d_x1); free(d_c1); free(d_f1);
    free(d_x2); free(d_c2); free(d_f2);
    free(coarse); free(fine);
}

// ============================================================
// Main
// ============================================================
int main(void) {
    printf("=== Nested MoE Router Backward Tests ===\n");
    printf("D_MODEL=%d, N_EXPERTS=%d, N_ACTIVE_EXPTS=%d\n",
           D_MODEL, N_EXPERTS, N_ACTIVE_EXPTS);
    printf("N_HYPERBOLIC_GROUPS=%d, N_EXPERTS_PER_GROUP=%d\n",
           N_HYPERBOLIC_GROUPS, N_EXPERTS_PER_GROUP);
    printf("R_coarse=%.1f, R_fine=%.1f, temp=%.2f\n\n",
           R_POINCARE_COARSE, R_POINCARE_FINE, HYPERBOLIC_TEMPERATURE);

    srand(42);  // deterministic seed

    test_basic_backward();
    test_null_guards();
    test_uniform_weight_gradient();
    test_deterministic();

    printf("\n=== Results: %d/%d tests passed ===\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
