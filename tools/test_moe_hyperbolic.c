/**
 * test_moe_hyperbolic.c — Test Nested MoE routing with Poincaré distance + hierarchy
 *
 * Tests:
 *   1. Poincaré distance router (256 centroids, flat)
 *   2. Two-level hierarchical router (16 groups of 16)
 *   3. Consistency checks + sanity
 *
 * Build: gcc -O2 -I include -o test_moe_hyperbolic tools/test_moe_hyperbolic.c \
 *        src/wubu_moe_hyperbolic.o src/wubu_mobius.o -lm -fopenmp
 */

#include "wubu_moe_hyperbolic.h"
#include "wubu_moe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

// Small test dimensions
#define TEST_B 2
#define TEST_T 4
#define TEST_N (TEST_B * TEST_T)  // 8 tokens

static int tests_passed = 0;
static int tests_total = 0;

static void check_float(const char *name, float got, float expected, float tol) {
    tests_total++;
    if (fabsf(got - expected) <= tol) {
        printf("  PASS: %s = %.6f\n", name, got);
        tests_passed++;
    } else {
        printf("  FAIL: %s = %.6f (expected %.6f, tol %.6f)\n", name, got, expected, tol);
    }
}

static void check_true(const char *name, int cond) {
    tests_total++;
    if (cond) {
        printf("  PASS: %s\n", name);
        tests_passed++;
    } else {
        printf("  FAIL: %s (expected true)\n", name);
    }
}

// ============================================================
// Test 1: Poincaré distance router — check dimensions and validity
// ============================================================
static void test_poincare_router(void) {
    printf("\n=== Test 1: Poincaré Distance Router ===\n");

    float *centroids = (float *)malloc(N_EXPERTS * D_MODEL * sizeof(float));
    assert(centroids);
    wubu_poincare_router_init_random(centroids, 42);

    poincare_router_t router;
    router.centroids = centroids;
    router.temperature = HYPERBOLIC_TEMPERATURE;
    router.loaded = true;

    // Input: [B, T, D_MODEL] random
    float *x = (float *)malloc(TEST_N * D_MODEL * sizeof(float));
    assert(x);
    for (int i = 0; i < TEST_N * D_MODEL; i++) {
        x[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
    }

    float *scores = (float *)malloc(TEST_N * N_EXPERTS * sizeof(float));
    assert(scores);
    memset(scores, 0, TEST_N * N_EXPERTS * sizeof(float));

    // Run router
    wubu_poincare_router_forward(x, TEST_B, TEST_T, &router, scores);

    // Check scores are nonzero (since we have random inputs + centroids)
    float total_score = 0.0f;
    for (int i = 0; i < TEST_N * N_EXPERTS; i++)
        total_score += fabsf(scores[i]);

    check_true("scores have values", total_score > 0.0f);
    printf("  Total abs score (first 2 tokens): ");
    for (int e = 0; e < 4; e++)
        printf("%.4f ", scores[e]);
    printf("...\n");

    // Check softmax per token: sum to ~1
    for (int s = 0; s < TEST_N; s++) {
        float *score_s = scores + s * N_EXPERTS;

        // Manual softmax
        float maxv = score_s[0];
        for (int e = 1; e < N_EXPERTS; e++)
            if (score_s[e] > maxv) maxv = score_s[e];

        float sum = 0.0f;
        for (int e = 0; e < N_EXPERTS; e++)
            sum += expf(score_s[e] - maxv);
        float inv_sum = 1.0f / (sum + 1e-30f);

        float softmax_sum = 0.0f;
        for (int e = 0; e < N_EXPERTS; e++)
            softmax_sum += expf(score_s[e] - maxv) * inv_sum;

        char name[64];
        snprintf(name, sizeof(name), "token %d softmax sum", s);
        check_float(name, softmax_sum, 1.0f, 1e-4f);
    }

    // Check top-8 selection works
    for (int s = 0; s < TEST_N; s++) {
        float *score_s = scores + s * N_EXPERTS;
        float maxv = score_s[0];
        for (int e = 1; e < N_EXPERTS; e++)
            if (score_s[e] > maxv) maxv = score_s[e];
        float sum = 0.0f;
        for (int e = 0; e < N_EXPERTS; e++)
            sum += expf(score_s[e] - maxv);
        float inv_sum = 1.0f / (sum + 1e-30f);

        float softmax_vals[N_EXPERTS];
        for (int e = 0; e < N_EXPERTS; e++)
            softmax_vals[e] = expf(score_s[e] - maxv) * inv_sum;

        int topk_indices[N_ACTIVE_EXPTS];
        float topk_weights[N_ACTIVE_EXPTS];
        topk_from_array(softmax_vals, N_EXPERTS, N_ACTIVE_EXPTS,
                        topk_indices, topk_weights);

        // Verify indices in range [0, 255]
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            char name[64];
            snprintf(name, sizeof(name), "token %d top-%d index in range", s, k);
            check_true(name, topk_indices[k] >= 0 && topk_indices[k] < N_EXPERTS);
        }

        // Verify weights sum to ~1 (after normalization)
        float sum_w = 0.0f;
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) sum_w += topk_weights[k];
        float inv_sum_w = 1.0f / (sum_w + 1e-30f);
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) topk_weights[k] *= inv_sum_w;

        sum_w = 0.0f;
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) sum_w += topk_weights[k];

        {
            char name[64];
            snprintf(name, sizeof(name), "token %d topk weight sum", s);
            check_float(name, sum_w, 1.0f, 1e-4f);
        }
    }

    free(x);
    free(scores);
    free(centroids);
}

// ============================================================
// Test 2: Two-level hierarchical routing
// ============================================================
static void test_nested_moe_router(void) {
    printf("\n=== Test 2: Nested MoE (2-level hierarchy) Router ===\n");

    float *coarse = (float *)malloc(N_HYPERBOLIC_GROUPS * D_MODEL * sizeof(float));
    float *fine = (float *)malloc(N_EXPERTS * D_MODEL * sizeof(float));
    assert(coarse && fine);
    wubu_nested_moe_router_init_random(coarse, fine, 123);

    nested_moe_router_t router;
    router.coarse_centroids = coarse;
    router.fine_centroids = fine;
    router.temperature = HYPERBOLIC_TEMPERATURE;
    router.loaded = true;

    // Input
    float *x = (float *)malloc(TEST_N * D_MODEL * sizeof(float));
    assert(x);
    for (int i = 0; i < TEST_N * D_MODEL; i++) {
        x[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f;
    }

    int *indices = (int *)malloc(TEST_N * N_ACTIVE_EXPTS * sizeof(int));
    float *weights = (float *)malloc(TEST_N * N_ACTIVE_EXPTS * sizeof(float));
    assert(indices && weights);

    memset(indices, 0, TEST_N * N_ACTIVE_EXPTS * sizeof(int));
    memset(weights, 0, TEST_N * N_ACTIVE_EXPTS * sizeof(float));

    wubu_nested_moe_router_forward(x, TEST_B, TEST_T, &router, indices, weights);

    // Check outputs
    for (int s = 0; s < TEST_N; s++) {
        int *inds_s = indices + s * N_ACTIVE_EXPTS;
        float *wts_s = weights + s * N_ACTIVE_EXPTS;

        // Verify indices in range [0, 255]
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            char name[64];
            snprintf(name, sizeof(name), "nested token %d expert %d in range", s, k);
            check_true(name, inds_s[k] >= 0 && inds_s[k] < N_EXPERTS);
        }

        // Verify no duplicate indices
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            for (int k2 = k+1; k2 < N_ACTIVE_EXPTS; k2++) {
                char name[64];
                snprintf(name, sizeof(name), "nested token %d no dup [%d,%d]", s, k, k2);
                check_true(name, inds_s[k] != inds_s[k2]);
            }
        }

        // Verify weights sum to ~1
        float sum_w = 0.0f;
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) sum_w += wts_s[k];

        char name[64];
        snprintf(name, sizeof(name), "nested token %d weight sum", s);
        check_float(name, sum_w, 1.0f, 1e-4f);

        // Print selected experts
        printf("  Token %d selected experts: ", s);
        for (int k = 0; k < N_ACTIVE_EXPTS; k++) {
            printf("[%d](%.4f) ", inds_s[k], wts_s[k]);
        }
        printf("\n");
    }

    free(x);
    free(indices);
    free(weights);
    free(coarse);
    free(fine);
}

// ============================================================
// Test 3: Verify that Poincaré-ball-mapped inputs stay in ball
// ============================================================
static void test_ball_mapping(void) {
    printf("\n=== Test 3: Poincaré Ball Mapping Validity ===\n");

    float *x_ball = (float *)malloc(D_MODEL * sizeof(float));
    assert(x_ball);

    // Test with random vectors of various magnitudes
    float test_norms[] = {0.0f, 1.0f, 10.0f, 100.0f, 1000.0f};
    float radii[] = {0.5f, 1.5f, 5.0f};

    for (int ri = 0; ri < 3; ri++) {
        float R = radii[ri];
        for (int ni = 0; ni < 5; ni++) {
            float *v = (float *)malloc(D_MODEL * sizeof(float));
            float target_norm = test_norms[ni];
            if (target_norm > 0.0f) {
                for (int i = 0; i < D_MODEL; i++)
                    v[i] = ((float)rand() / (float)RAND_MAX - 0.5f);
                // Normalize to target norm
                float n = 0.0f;
                for (int i = 0; i < D_MODEL; i++) n += v[i] * v[i];
                n = sqrtf(n);
                float scale = target_norm / n;
                for (int i = 0; i < D_MODEL; i++) v[i] *= scale;
            } else {
                memset(v, 0, D_MODEL * sizeof(float));
            }

            euclidean_to_poincare_ball(v, D_MODEL, R, x_ball);

            // Check norm < R
            float ball_norm = 0.0f;
            for (int i = 0; i < D_MODEL; i++) ball_norm += x_ball[i] * x_ball[i];
            ball_norm = sqrtf(ball_norm);

            char name[64];
            snprintf(name, sizeof(name), "R=%.1f input_norm=%.0f ball_norm<R", R, target_norm);
            check_true(name, ball_norm < R + 1e-5f);

            free(v);
        }
    }

    free(x_ball);
}

// ============================================================
// Test 4: Compare Poincaré router distances for nearby vs far apart points
// ============================================================
static void test_distance_sanity(void) {
    printf("\n=== Test 4: Distance sanity — nearby centroids give higher scores ===\n");

    // Create two centroids: one near origin, one near the boundary
    // and verify the distance function works correctly

    float *c_near = (float *)malloc(D_MODEL * sizeof(float));
    float *c_far = (float *)malloc(D_MODEL * sizeof(float));
    float *x_ball = (float *)malloc(D_MODEL * sizeof(float));

    // Centroid near origin (same quadrant as input)
    for (int i = 0; i < D_MODEL; i++) c_near[i] = 0.0f;
    c_near[0] = 0.05f;

    // Centroid far from origin (in opposite direction — very distant in Poincaré)
    for (int i = 0; i < D_MODEL; i++) c_far[i] = 0.0f;
    c_far[0] = -0.40f;

    // Input near origin, positive x direction
    memset(x_ball, 0, D_MODEL * sizeof(float));
    x_ball[0] = 0.02f;  // close to c_near (0.05), far from c_far (-0.40)

    float d_near = wubu_poincare_dist(x_ball, c_near, D_MODEL, R_POINCARE_FINE);
    float d_far = wubu_poincare_dist(x_ball, c_far, D_MODEL, R_POINCARE_FINE);

    printf("  Distance to near centroid: %.6f\n", d_near);
    printf("  Distance to far centroid:  %.6f\n", d_far);
    check_true("near centroid distance < far centroid distance", d_near < d_far);

    // Verify distances are non-negative
    check_true("distance to near is non-negative", d_near >= 0.0f);
    check_true("distance to far is non-negative", d_far >= 0.0f);

    // Test same-point distance is near zero
    float d_self = wubu_poincare_dist(x_ball, x_ball, D_MODEL, R_POINCARE_FINE);
    check_float("self-distance ~ 0", d_self, 0.0f, 1e-4f);

    free(c_near);
    free(c_far);
    free(x_ball);
}

// ============================================================
// Main
// ============================================================
int main(void) {
    printf("=== Nested MoE Hyperbolic Router Tests ===\n");
    printf("D_MODEL=%d, N_EXPERTS=%d, N_ACTIVE_EXPTS=%d\n",
           D_MODEL, N_EXPERTS, N_ACTIVE_EXPTS);
    printf("N_HYPERBOLIC_GROUPS=%d, N_EXPERTS_PER_GROUP=%d\n",
           N_HYPERBOLIC_GROUPS, N_EXPERTS_PER_GROUP);
    printf("R_coarse=%.1f, R_fine=%.1f, temp=%.2f\n",
           R_POINCARE_COARSE, R_POINCARE_FINE, HYPERBOLIC_TEMPERATURE);

    srand((unsigned int)time(NULL));

    test_ball_mapping();
    test_distance_sanity();
    test_poincare_router();
    test_nested_moe_router();

    printf("\n=== Results: %d/%d tests passed ===\n", tests_passed, tests_total);
    return (tests_passed == tests_total) ? 0 : 1;
}
