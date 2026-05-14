// tools/test_tst.c — Test TST Token Superposition Training
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "wubu_tst.h"

// ============================================================
// Utility: check two floats are approximately equal
// ============================================================
static int approx_eq(float a, float b, float eps) {
    return fabsf(a - b) < eps;
}

// ============================================================
// Test 1: Bag embeddings basic
// ============================================================
static int test_bag_embeddings(void) {
    printf("--- Test 1: Bag embeddings basic ---\n");
    printf("  SKIP (replaced by Test 2)\n");
    return 1;
}

// ============================================================
// Test 2: Bag embeddings (cleaned up dimensions)
// ============================================================
static int test_bag_embeddings_clean(void) {
    printf("--- Test 2: Bag embeddings (clean dims) ---\n");
    
    int B = 1, T = 8, D = 3, s = 4;
    // Tokens: t0=(1,0,0), t1=(0,1,0), t2=(0,0,1), t3=(0,0,0)
    //         t4=(2,0,0), t5=(0,2,0), t6=(0,0,2), t7=(0,0,0)
    float emb[] = {
        1,0,0,  0,1,0,  0,0,1,  0,0,0,
        2,0,0,  0,2,0,  0,0,2,  0,0,0
    };
    float bagged[1 * 2 * 3];  // B * T_out * D = 1*2*3
    memset(bagged, 0, sizeof(bagged));
    
    tst_bag_embeddings(emb, bagged, B, T, D, s);
    // Bag 0 (t0..t3): avg = (0.25, 0.25, 0.25)
    // Bag 1 (t4..t7): avg = (0.5, 0.5, 0.5)
    
    float exp0[] = {0.25f, 0.25f, 0.25f};
    float exp1[] = {0.5f, 0.5f, 0.5f};
    
    int ok = 1;
    for (int d = 0; d < D; d++) {
        if (!approx_eq(bagged[d], exp0[d], 1e-5f)) ok = 0;
        if (!approx_eq(bagged[3 + d], exp1[d], 1e-5f)) ok = 0;
    }
    
    if (ok) {
        printf("  PASS\n");
    } else {
        printf("  FAIL: got (%.4f,%.4f,%.4f) (%.4f,%.4f,%.4f)\n",
               bagged[0], bagged[1], bagged[2],
               bagged[3], bagged[4], bagged[5]);
    }
    return ok;
}

// ============================================================
// Test 3: Target preparation
// ============================================================
static int test_target_prep(void) {
    printf("--- Test 3: Target preparation ---\n");
    
    int B = 1, T = 15, s = 8;
    // Tokens 0..14
    int token_ids[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    int targets[1 * 1 * 8];  // B * T_out * s, T_out = (15-8+1)/8 = 1
    memset(targets, 0xFF, sizeof(targets));  // fill with -1
    
    int T_out = tst_prepare_targets(token_ids, targets, B, T, s);
    
    if (T_out != 1) {
        printf("  FAIL: expected T_out=1, got %d\n", T_out);
        return 0;
    }
    
    // Shift left by s-1=7: seq = token_ids[7:] = [7,8,9,10,11,12,13,14]
    // Bag 0: [7,8,9,10,11,12,13,14]
    int expected[] = {7, 8, 9, 10, 11, 12, 13, 14};
    int ok = 1;
    for (int k = 0; k < s; k++) {
        if (targets[k] != expected[k]) {
            ok = 0;
            printf("  FAIL: target[%d] = %d, expected %d\n", k, targets[k], expected[k]);
        }
    }
    
    if (ok) printf("  PASS\n");
    return ok;
}

// ============================================================
// Test 4: Target preparation with multiple bags
// ============================================================
static int test_target_prep_multi(void) {
    printf("--- Test 4: Target preparation (multi-bag) ---\n");
    
    int B = 1, T = 23, s = 8;
    int token_ids[23];
    for (int i = 0; i < T; i++) token_ids[i] = i;
    
    int targets[1 * 2 * 8];  // T_out = (23-8+1)/8 = 2
    memset(targets, 0xFF, sizeof(targets));
    
    int T_out = tst_prepare_targets(token_ids, targets, B, T, s);
    
    if (T_out != 2) {
        printf("  FAIL: expected T_out=2, got %d\n", T_out);
        return 0;
    }
    
    // Shift left by 7: [7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    // Bag 0: [7,8,9,10,11,12,13,14]
    // Bag 1: [15,16,17,18,19,20,21,22]
    int exp0[] = {7,8,9,10,11,12,13,14};
    int exp1[] = {15,16,17,18,19,20,21,22};
    
    int ok = 1;
    for (int k = 0; k < s; k++) {
        if (targets[k] != exp0[k]) { ok = 0; printf("  FAIL bag0[%d]=%d exp %d\n", k, targets[k], exp0[k]); }
        if (targets[8 + k] != exp1[k]) { ok = 0; printf("  FAIL bag1[%d]=%d exp %d\n", k, targets[8+k], exp1[k]); }
    }
    
    if (ok) printf("  PASS\n");
    return ok;
}

// ============================================================
// Test 5: Single CE loss
// ============================================================
static int test_single_ce(void) {
    printf("--- Test 5: Single cross-entropy ---\n");
    
    // Simple case: logits = [0, 5, 0], label = 1
    // softmax: exp(0)/sum = 0.0067, exp(5)/sum = 0.9866, exp(0)/sum = 0.0067
    // CE = -log(0.9866) ≈ 0.0135
    float logits[] = {0.0f, 5.0f, 0.0f};
    float ce = tst_cross_entropy(logits, 3, 1);
    
    float expected = -logf(expf(5) / (expf(0) + expf(5) + expf(0)));
    
    if (approx_eq(ce, expected, 1e-4f)) {
        printf("  PASS (CE=%.6f)\n", ce);
        return 1;
    } else {
        printf("  FAIL: CE=%.6f, expected ~%.6f\n", ce, expected);
        return 0;
    }
}

// ============================================================
// Test 6: MCE loss with synthetic data
// ============================================================
static int test_mce_loss(void) {
    printf("--- Test 6: MCE loss ---\n");
    
    int B = 1, T_out = 2, V = 5, s = 3;
    
    // Two predictions, 5 vocab, 3 targets each
    // Prediction 0: logits = [5, 0, 0, 0, 0]  (strongly predicts class 0)
    //   targets: [0, 0, 0]  (all correct → low CE for each)
    // Prediction 1: logits = [0, 5, 0, 0, 0]  (strongly predicts class 1)
    //   targets: [1, 2, 1]  (one wrong → higher CE for target 2)
    float logits[] = {
        5.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 5.0f, 0.0f, 0.0f, 0.0f
    };
    int targets[] = {
        0, 0, 0,
        1, 2, 1
    };
    
    float loss;
    bool ok = tst_compute_mce_loss(logits, targets, B, T_out, V, s, &loss);
    
    if (!ok) {
        printf("  FAIL: returned false\n");
        return 0;
    }
    
    // Manual calculation
    // CE for pred 0, target 0: -log(softmax(0)[0]) = -log(exp(5)/(exp(5)+4*exp(0)))
    float s0_denom = expf(5.0f) + 4.0f * expf(0.0f);
    float ce00 = -logf(expf(5.0f) / s0_denom);
    // CE for pred 0, target 0 (second): same
    // CE for pred 0, target 0 (third): same
    float row0_loss = (ce00 * 3.0f) / 3.0f;  // = ce00
    
    // CE for pred 1, target 1: -log(softmax(1)[1]) = -log(exp(5)/(exp(5)+4*exp(0)))
    // CE for pred 1, target 2: -log(softmax(1)[2]) = -log(exp(0)/(exp(5)+4*exp(0)))
    float s1_denom = expf(5.0f) + 4.0f * expf(0.0f);
    float ce11 = -logf(expf(5.0f) / s1_denom);  // same as ce00
    float ce12 = -logf(expf(0.0f) / s1_denom);   // higher loss
    float row1_loss = (ce11 + ce12 + ce11) / 3.0f;
    
    float expected = (row0_loss + row1_loss) / 2.0f;
    
    if (approx_eq(loss, expected, 1e-4f)) {
        printf("  PASS (loss=%.6f)\n", loss);
        return 1;
    } else {
        printf("  FAIL: loss=%.6f, expected ~%.6f\n", loss, expected);
        printf("  Row 0 loss=%.6f, Row 1 loss=%.6f\n", row0_loss, row1_loss);
        return 0;
    }
}

// ============================================================
// Test 7: MCE loss backward — check gradient correctness
// ============================================================
static int test_mce_backward(void) {
    printf("--- Test 7: MCE backward ---\n");
    
    int B = 1, T_out = 2, V = 5, s = 3;
    
    float logits[] = {
        5.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 5.0f, 0.0f, 0.0f, 0.0f
    };
    int targets[] = {
        0, 0, 0,
        1, 2, 1
    };
    int N = B * T_out;
    float *d_logits = (float *)calloc(N * V, sizeof(float));
    
    tst_mce_loss_backward(logits, targets, B, T_out, V, s, d_logits);
    
    // Verify: gradient sum for each row = (1/s - 1)
    // Because ∂loss/∂logits[n,j] = (1/s) * [softmax[n,j] - Σ_k δ(j==tgt[n,k])]
    // Sum_j = (1/s) * [1 - s] = 1/s - 1
    float expected_sum = 1.0f / (float)s - 1.0f;
    int ok = 1;
    for (int n = 0; n < N; n++) {
        float row_sum = 0.0f;
        for (int j = 0; j < V; j++) {
            row_sum += d_logits[n * V + j];
        }
        if (!approx_eq(row_sum, expected_sum, 1e-5f)) {
            printf("  FAIL: row %d gradient sum = %.6f (expected %.6f)\n",
                   n, row_sum, expected_sum);
            ok = 0;
        }
    }
    
    if (ok) printf("  PASS (gradient sum = 1/s - 1 = %.4f per row)\n", expected_sum);
    free(d_logits);
    return ok;
}

// ============================================================
// Test 8: End-to-end TST superposition phase workflow
// ============================================================
static int test_e2e(void) {
    printf("--- Test 8: End-to-end TST superposition ---\n");
    
    // Use T such that T % s == 0 for bagging
    int B = 2, T = 16, D = 8, V = 10, s = 4;
    // Bagged count: T/s = 4
    // Target bags: (T-s+1)/s = 13/4 = 3 complete target bags
    
    // Create synthetic embeddings and token IDs
    int total_emb = B * T * D;
    float *emb = (float *)malloc(total_emb * sizeof(float));
    int *token_ids = (int *)malloc(B * T * sizeof(int));
    
    srand(42);
    for (int i = 0; i < total_emb; i++) emb[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < B * T; i++) token_ids[i] = rand() % V;
    
    // Step 1: Bag embeddings
    int T_out = T / s;
    float *bagged = (float *)malloc(B * T_out * D * sizeof(float));
    tst_bag_embeddings(emb, bagged, B, T, D, s);
    printf("  Bagged embeddings: [%d, %d, %d]\n", B, T_out, D);
    
    // Step 2: Prepare TST targets
    int *tst_targets = (int *)malloc(B * (T / s) * s * sizeof(int));
    int T_tgt = tst_prepare_targets(token_ids, tst_targets, B, T, s);
    printf("  TST targets: %d bags of %d\n", T_tgt, s);
    
    // Step 3: Simulate forward pass — logits for each bagged position
    int N_pred = B * T_out;
    float *logits = (float *)malloc(N_pred * V * sizeof(float));
    for (int i = 0; i < N_pred * V; i++) logits[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    
    // Step 4: Compute MCE loss on first T_tgt bags (rest would be masked in training)
    int N_loss = B * T_tgt;
    float mce_loss;
    bool ok = tst_compute_mce_loss(logits, tst_targets, B, T_tgt, V, s, &mce_loss);
    
    if (!ok || mce_loss < 0.0f || mce_loss > 10.0f) {
        printf("  FAIL: suspicious MCE loss = %.6f\n", mce_loss);
        free(emb); free(token_ids); free(bagged);
        free(tst_targets); free(logits);
        return 0;
    }
    
    printf("  MCE loss = %.6f (reasonable)\n", mce_loss);
    
    // Step 5: Compute gradients on the same subset
    float *d_logits = (float *)calloc(N_pred * V, sizeof(float));
    tst_mce_loss_backward(logits, tst_targets, B, T_tgt, V, s, d_logits);
    
    // Verify gradient sum = (1/s - 1) for each row that had targets
    float expected_sum = 1.0f / (float)s - 1.0f;
    int grad_ok = 1;
    for (int n = 0; n < N_loss; n++) {
        float row_sum = 0.0f;
        for (int j = 0; j < V; j++) row_sum += d_logits[n * V + j];
        if (!approx_eq(row_sum, expected_sum, 1e-4f)) {
            printf("  FAIL: row %d grad sum = %.6f (expected %.6f)\n",
                   n, row_sum, expected_sum);
            grad_ok = 0;
        }
    }
    // Remaining rows (N_loss..N_pred-1) should be untouched (target count didn't cover them)
    
    printf("  Gradient: %s\n", grad_ok ? "PASS" : "FAIL");
    
    free(emb);
    free(token_ids);
    free(bagged);
    free(tst_targets);
    free(logits);
    free(d_logits);
    
    int result = ok && grad_ok;
    printf("  %s\n", result ? "PASS" : "FAIL");
    return result;
}

// ============================================================
// Main
// ============================================================
int main(void) {
    printf("=== TST Token Superposition Training Tests ===\n\n");
    
    int pass = 0, total = 0;
    
    total++; pass += test_bag_embeddings();
    total++; pass += test_bag_embeddings_clean();
    total++; pass += test_target_prep();
    total++; pass += test_target_prep_multi();
    total++; pass += test_single_ce();
    total++; pass += test_mce_loss();
    total++; pass += test_mce_backward();
    total++; pass += test_e2e();
    
    printf("\n=== Results: %d/%d passed ===\n", pass, total);
    return (pass == total) ? 0 : 1;
}
