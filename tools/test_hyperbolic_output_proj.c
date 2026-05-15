/**
 * test_hyperbolic_output_proj.c — Validate hyperbolic output projection.
 *
 * Tests:
 * 1. Forward output is finite (no NaN/Inf)
 * 2. Forward output in expected range
 * 3. Backward gradients are non-zero
 * 4. Backward with and without saved l_ball produce consistent gradients
 * 5. Gradient accumulation works (multiple backward calls)
 */

#include "wubu_hyperbolic_output_proj.h"
#include "wubu_mobius.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Default dimensions matching the bytropix config
#define D_MODEL 2048
#define VOCAB_SMALL 4096   // small vocab for fast testing
#define N_BATCH 4          // batch * seqlen = 4 samples

// Compute max absolute value
static float max_abs(const float *data, int64_t n) {
    float m = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float v = fabsf(data[i]);
        if (v > m) m = v;
    }
    return m;
}

// Check all values are finite
static int all_finite(const float *data, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        if (!isfinite(data[i])) return 0;
    }
    return 1;
}

// Max absolute difference between two arrays
static float max_diff(const float *a, const float *b, int64_t n) {
    float m = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float v = fabsf(a[i] - b[i]);
        if (v > m) m = v;
    }
    return m;
}

int main() {
    int N = N_BATCH, D = D_MODEL, V = VOCAB_SMALL;
    float R_hidden = 2.0f;
    float R_logit  = 5.0f;

    printf("=== Hyperbolic Output Projection Test ===\n");
    printf("N=%d, D=%d, V=%d, R_hidden=%.1f, R_logit=%.1f\n\n",
           N, D, V, R_hidden, R_logit);

    // ---- Allocate inputs ----
    // Euclidean hidden states (simulating output of final rms_norm)
    float *hidden = (float *)malloc((int64_t)N * D * sizeof(float));
    // Weight matrix [V, D]
    float *W = (float *)malloc((int64_t)V * D * sizeof(float));
    // Bias [V]
    float *b = (float *)malloc(V * sizeof(float));

    if (!hidden || !W || !b) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    // Initialize hidden: small random Euclidean values
    for (int64_t i = 0; i < (int64_t)N * D; i++) {
        hidden[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    // Initialize weight: small random, He-like init
    float w_scale = sqrtf(2.0f / D);
    for (int64_t i = 0; i < (int64_t)V * D; i++) {
        W[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * w_scale;
    }

    // Initialize bias: small
    for (int i = 0; i < V; i++) {
        b[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    }

    // ---- Allocate outputs ----
    float *logits      = (float *)malloc((int64_t)N * V * sizeof(float));
    float *h_ball      = (float *)malloc((int64_t)N * D * sizeof(float));
    float *l_ball      = (float *)malloc((int64_t)N * V * sizeof(float));
    // For backward-all-saved mode
    float *d_hidden    = (float *)calloc((int64_t)N * D, sizeof(float));
    float *d_W         = (float *)calloc((int64_t)V * D, sizeof(float));
    float *d_b         = (float *)calloc(V, sizeof(float));
    // For backward-recompute mode  
    float *d_hidden2   = (float *)calloc((int64_t)N * D, sizeof(float));
    float *d_W2        = (float *)calloc((int64_t)V * D, sizeof(float));
    float *d_b2        = (float *)calloc(V, sizeof(float));

    // Upstream gradient (d_logits) — simulate softmax CE gradient
    float *d_logits    = (float *)malloc((int64_t)N * V * sizeof(float));

    if (!logits || !h_ball || !l_ball || !d_hidden || !d_W || !d_b ||
        !d_hidden2 || !d_W2 || !d_b2 || !d_logits) {
        fprintf(stderr, "Output allocation failed\n");
        return 1;
    }

    // ---- TEST 1: Forward with saved intermediates ----
    printf("--- Test 1: Forward (save h_ball + l_ball) ---\n");
    wubu_hyperbolic_output_proj_forward(
        hidden, N, D, V, W, b, R_hidden, R_logit, logits, h_ball, l_ball);

    int t1_finite  = all_finite(logits, (int64_t)N * V);
    int t1_h_finite = all_finite(h_ball, (int64_t)N * D);
    int t1_l_finite = all_finite(l_ball, (int64_t)N * V);
    float t1_max   = max_abs(logits, (int64_t)N * V);
    float t1_h_max  = max_abs(h_ball, (int64_t)N * D);

    // Check h_ball is in ball (||h_ball|| < R_hidden)
    int h_in_ball = 1;
    for (int i = 0; i < N; i++) {
        float n = wubu_norm(h_ball + (int64_t)i * D, D);
        if (n >= R_hidden * 0.999f) { h_in_ball = 0; break; }
    }

    // Check l_ball is in ball (||l_ball|| < R_logit)
    int l_in_ball = 1;
    for (int i = 0; i < N; i++) {
        float n = wubu_norm(l_ball + (int64_t)i * V, V);
        if (n >= R_logit * 0.999f) { l_in_ball = 0; break; }
    }

    printf("  logits finite:          %s\n", t1_finite  ? "✓ PASS" : "✗ FAIL");
    printf("  h_ball finite:          %s\n", t1_h_finite ? "✓ PASS" : "✗ FAIL");
    printf("  l_ball finite:          %s\n", t1_l_finite ? "✓ PASS" : "✗ FAIL");
    printf("  h_ball ||·|| < R_hidden: %s\n", h_in_ball  ? "✓ PASS" : "✗ FAIL");
    printf("  l_ball ||·|| < R_logit:  %s\n", l_in_ball  ? "✓ PASS" : "✗ FAIL");
    printf("  logits max_abs:          %.6e\n", t1_max);
    printf("  h_ball max_abs:          %.6e (R_hidden=%.1f)\n", t1_h_max, R_hidden);

    int t1_ok = t1_finite && t1_h_finite && t1_l_finite && h_in_ball && l_in_ball;
    printf("  => %s\n\n", t1_ok ? "PASSED" : "FAILED");

    // ---- TEST 2: Forward without saving intermediates (NULL h_ball, l_ball) ----
    printf("--- Test 2: Forward (no saving, NULL buffers) ---\n");
    float *logits2 = (float *)malloc((int64_t)N * V * sizeof(float));
    if (!logits2) { fprintf(stderr, "logits2 alloc failed\n"); return 1; }
    wubu_hyperbolic_output_proj_forward(
        hidden, N, D, V, W, b, R_hidden, R_logit, logits2, NULL, NULL);

    int t2_finite = all_finite(logits2, (int64_t)N * V);
    float t2_diff  = max_diff(logits, logits2, (int64_t)N * V);
    printf("  logits finite:    %s\n", t2_finite ? "✓ PASS" : "✗ FAIL");
    printf("  consistency diff: %.6e %s\n", t2_diff,
           t2_diff < 1e-6f ? "✓ PASS" : "⚠ small diff (float rounding)");
    int t2_ok = t2_finite && t2_diff < 1e-5f;
    printf("  => %s\n\n", t2_ok ? "PASSED" : "FAILED");
    free(logits2);

    // ---- Set up d_logits (upstream gradient) ----
    // Simulate softmax CE gradient: logits - one_hot(target)
    // Just use random targets here
    memset(d_logits, 0, (int64_t)N * V * sizeof(float));
    for (int i = 0; i < N; i++) {
        int tgt = rand() % V;
        // dL/dlogit = softmax(logit) - one_hot(tgt)
        // Approximate: just copy logits and subtract 1 at target
        memcpy(d_logits + (int64_t)i * V, logits + (int64_t)i * V, V * sizeof(float));
        d_logits[(int64_t)i * V + tgt] -= 1.0f;
    }

    // ---- TEST 3: Backward with saved intermediates ----
    printf("--- Test 3: Backward (saved h_ball + l_ball) ---\n");
    wubu_hyperbolic_output_proj_backward(
        hidden, N, D, V, W, b, R_hidden, R_logit,
        h_ball, l_ball, d_logits,
        d_hidden, d_W, d_b);

    float t3_max_dh = max_abs(d_hidden, (int64_t)N * D);
    float t3_max_dW = max_abs(d_W,     (int64_t)V * D);
    float t3_max_db = max_abs(d_b,     V);
    int t3_dh_finite = all_finite(d_hidden, (int64_t)N * D);
    int t3_dW_finite = all_finite(d_W,     (int64_t)V * D);
    int t3_db_finite = all_finite(d_b,     V);
    int t3_dh_nonzero = t3_max_dh > 0;
    int t3_dW_nonzero = t3_max_dW > 0;
    int t3_db_nonzero = t3_max_db > 0;

    printf("  d_hidden finite:  %s  max=%.6e  %s\n",
           t3_dh_finite ? "✓" : "✗", t3_max_dh,
           t3_dh_nonzero ? "✓ non-zero" : "✗ zero");
    printf("  d_W finite:       %s  max=%.6e  %s\n",
           t3_dW_finite ? "✓" : "✗", t3_max_dW,
           t3_dW_nonzero ? "✓ non-zero" : "✗ zero");
    printf("  d_b finite:       %s  max=%.6e  %s\n",
           t3_db_finite ? "✓" : "✗", t3_max_db,
           t3_db_nonzero ? "✓ non-zero" : "✗ zero");

    int t3_ok = t3_dh_finite && t3_dW_finite && t3_db_finite &&
                t3_dh_nonzero && t3_dW_nonzero && t3_db_nonzero;
    printf("  => %s\n\n", t3_ok ? "PASSED" : "FAILED");

    // ---- TEST 4: Backward with recomputed intermediates (l_ball=NULL) ----
    printf("--- Test 4: Backward (recompute l_ball, memory-efficient) ---\n");
    wubu_hyperbolic_output_proj_backward(
        hidden, N, D, V, W, b, R_hidden, R_logit,
        h_ball, NULL, d_logits,  // l_ball=NULL → recompute
        d_hidden2, d_W2, d_b2);

    float t4_max_dh = max_abs(d_hidden2, (int64_t)N * D);
    float t4_max_dW = max_abs(d_W2,     (int64_t)V * D);
    float t4_max_db = max_abs(d_b2,     V);
    int t4_finite = all_finite(d_hidden2, (int64_t)N * D) &&
                    all_finite(d_W2,     (int64_t)V * D) &&
                    all_finite(d_b2,     V);
    int t4_nonzero = t4_max_dh > 0 && t4_max_dW > 0 && t4_max_db > 0;

    // Compare with saved-mode backward
    float t4_dh_diff = max_diff(d_hidden, d_hidden2, (int64_t)N * D);
    float t4_dW_diff = max_diff(d_W,     d_W2,     (int64_t)V * D);
    float t4_db_diff = max_diff(d_b,     d_b2,     V);

    printf("  finite:       %s\n", t4_finite ? "✓ PASS" : "✗ FAIL");
    printf("  non-zero:     %s\n", t4_nonzero ? "✓ PASS" : "✗ FAIL");
    printf("  d_hidden diff: %.6e\n", t4_dh_diff);
    printf("  d_W diff:      %.6e\n", t4_dW_diff);
    printf("  d_b diff:      %.6e\n", t4_db_diff);

    // Gradients should be identical (both paths compute the same math)
    int t4_consistent = t4_dh_diff < 1e-5f && t4_dW_diff < 1e-5f && t4_db_diff < 1e-5f;
    int t4_ok = t4_finite && t4_nonzero && t4_consistent;
    printf("  consistent:   %s\n", t4_consistent ? "✓ PASS" : "✗ FAIL");
    printf("  => %s\n\n", t4_ok ? "PASSED" : "FAILED");

    // ---- TEST 5: Gradient independence (separate calls, different inputs) ----
    printf("--- Test 5: Gradient independence (different inputs → different grads) ---\n");
    // Create a second set of hidden states with different values
    float *hidden2 = (float *)malloc((int64_t)N * D * sizeof(float));
    for (int64_t i = 0; i < (int64_t)N * D; i++) {
        hidden2[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    float *logits_alt = (float *)malloc((int64_t)N * V * sizeof(float));
    float *h_ball_alt  = (float *)malloc((int64_t)N * D * sizeof(float));
    float *l_ball_alt  = (float *)malloc((int64_t)N * V * sizeof(float));
    wubu_hyperbolic_output_proj_forward(
        hidden2, N, D, V, W, b, R_hidden, R_logit, logits_alt, h_ball_alt, l_ball_alt);

    // d_logits_alt: different targets
    float *d_logits_alt = (float *)malloc((int64_t)N * V * sizeof(float));
    memset(d_logits_alt, 0, (int64_t)N * V * sizeof(float));
    for (int i = 0; i < N; i++) {
        int tgt = rand() % V;
        memcpy(d_logits_alt + (int64_t)i * V, logits_alt + (int64_t)i * V, V * sizeof(float));
        d_logits_alt[(int64_t)i * V + tgt] -= 1.0f;
    }

    // Run backward on the alternate hidden
    float *d_W_alt = (float *)calloc((int64_t)V * D, sizeof(float));
    float *d_b_alt = (float *)calloc(V, sizeof(float));
    wubu_hyperbolic_output_proj_backward(
        hidden2, N, D, V, W, b, R_hidden, R_logit,
        h_ball_alt, l_ball_alt, d_logits_alt,
        NULL, d_W_alt, d_b_alt);

    // Compare: gradients from different inputs should differ
    float t5_dW_diff = max_diff(d_W, d_W_alt, (int64_t)V * D);
    float t5_db_diff = max_diff(d_b, d_b_alt, V);
    int t5_different = t5_dW_diff > 1e-6f && t5_db_diff > 1e-6f;
    printf("  d_W diff (different inputs): %.6e %s\n",
           t5_dW_diff, t5_dW_diff > 1e-6f ? "✓ (different)" : "✗ (same)");
    printf("  d_b diff (different inputs): %.6e %s\n",
           t5_db_diff, t5_db_diff > 1e-6f ? "✓ (different)" : "✗ (same)");

    // Also verify d_hidden accumulation: running backward twice on same inputs
    // adds to existing d_hidden (since mobius_linear_backward ADDS to d_x)
    float *d_hidden_accum = (float *)calloc((int64_t)N * D, sizeof(float));
    wubu_hyperbolic_output_proj_backward(
        hidden, N, D, V, W, b, R_hidden, R_logit,
        h_ball, l_ball, d_logits,
        d_hidden_accum, NULL, NULL);
    float t5_dh_pass1 = max_abs(d_hidden_accum, (int64_t)N * D);
    wubu_hyperbolic_output_proj_backward(
        hidden, N, D, V, W, b, R_hidden, R_logit,
        h_ball, l_ball, d_logits,
        d_hidden_accum, NULL, NULL);
    float t5_dh_pass2 = max_abs(d_hidden_accum, (int64_t)N * D);

    // After two calls, max should be approximately 2x (within float tolerance)
    float t5_dh_ratio = t5_dh_pass2 / (t5_dh_pass1 + 1e-30f);
    int t5_accum_ok = t5_dh_ratio > 1.9f && t5_dh_ratio < 2.1f;
    printf("  d_hidden max pass 1: %.6e\n", t5_dh_pass1);
    printf("  d_hidden max pass 2: %.6e (ratio=%.2f) %s\n",
           t5_dh_pass2, t5_dh_ratio, t5_accum_ok ? "✓ 2x" : "✗");

    int t5_ok = t5_different && t5_accum_ok;
    printf("  => %s\n\n", t5_ok ? "PASSED" : "FAILED");

    free(hidden2);
    free(logits_alt);
    free(h_ball_alt);
    free(l_ball_alt);
    free(d_logits_alt);
    free(d_W_alt);
    free(d_b_alt);
    free(d_hidden_accum);

    // ---- Summary ----
    printf("=== SUMMARY ===\n");
    printf("Test 1 (forward, saved): %s\n", t1_ok ? "PASSED ✓" : "FAILED ✗");
    printf("Test 2 (forward, no-save): %s\n", t2_ok ? "PASSED ✓" : "FAILED ✗");
    printf("Test 3 (backward, saved): %s\n", t3_ok ? "PASSED ✓" : "FAILED ✗");
    printf("Test 4 (backward, recompute): %s\n", t4_ok ? "PASSED ✓" : "FAILED ✗");
    printf("Test 5 (accumulation): %s\n", t5_ok ? "PASSED ✓" : "FAILED ✗");

    int all_ok = t1_ok && t2_ok && t3_ok && t4_ok && t5_ok;
    printf("\n%s\n", all_ok ? "ALL TESTS PASSED ✓" : "SOME TESTS FAILED ✗");

    // ---- Cleanup ----
    free(hidden);
    free(W);
    free(b);
    free(logits);
    free(h_ball);
    free(l_ball);
    free(d_hidden);
    free(d_W);
    free(d_b);
    free(d_hidden2);
    free(d_W2);
    free(d_b2);
    free(d_logits);

    return all_ok ? 0 : 1;
}
