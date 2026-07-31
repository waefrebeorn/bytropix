/* Test MoD layer skip (doc 017).
 *
 * Tests two paths:
 *   1. Gate < 0.5 → skip: y = x (passthrough, no F(x) computed)
 *   2. Gate >= 1.0 → no skip: y = x + F(x) (residual)
 *   3. Floor: last N layers always run regardless of gate
 */
#include "wubu_layer_skip.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main(void) {
    int n_tokens = 1, D = 4, total_layers = 10, floor = 2;

    /* --- Test 1: Skip path (gate weight forces low gate) ---
     * gate_weight = large negative → σ(gate) ≈ 0 → skip.
     */
    float x_skip[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float gate_w_skip[4] = {-10.0f, -10.0f, -10.0f, -10.0f};
    float F_skip[4] = {100.0f, 200.0f, 300.0f, 400.0f}; /* not used */
    float y_skip[4];
    memcpy(y_skip, F_skip, sizeof(y_skip));  /* pre-fill with F(x) */

    wubu_layer_skip_forward(x_skip, gate_w_skip, y_skip, n_tokens, D,
            total_layers, 0, floor, true, 0.5f, 0.5f);

    /* If skipped, y should be x (passthrough), NOT x + F(x). */
    for (int i = 0; i < D; i++) {
        if (fabsf(y_skip[i] - x_skip[i]) > 1e-5f) {
            fprintf(stderr, "SKIP path FAILED: y[%d]=%.2f expected %.2f\n",
                    i, (double)y_skip[i], (double)x_skip[i]);
            return 1;
        }
    }

    /* --- Test 2: No-skip path (gate weight forces high gate) ---
     * gate_weight = large positive → σ(gate) ≈ 1 → no skip.
     */
    float x_run[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float gate_w_run[4] = {10.0f, 10.0f, 10.0f, 10.0f};
    float F_run[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    float y_run[4];
    memcpy(y_run, F_run, sizeof(y_run));  /* pre-fill with F(x) */

    wubu_layer_skip_forward(x_run, gate_w_run, y_run, n_tokens, D,
            total_layers, 0, floor, true, 0.5f, 0.5f);

    /* If not skipped, y should be x + F(x) (residual). */
    for (int i = 0; i < D; i++) {
        float expected = x_run[i] + F_run[i];
        if (fabsf(y_run[i] - expected) > 1e-5f) {
            fprintf(stderr, "RUN path FAILED: y[%d]=%.2f expected %.2f\n",
                    i, (double)y_run[i], (double)expected);
            return 1;
        }
    }

    /* --- Test 3: Floor enforcement (last 2 layers always run) --- */
    if (wubu_layer_skip_verify_floor(total_layers, total_layers - floor, floor)) {
        fprintf(stderr, "Floor FAILED: layer %d should be in floor (not skippable)\n",
                total_layers - floor);
        return 1;
    }
    if (!wubu_layer_skip_verify_floor(total_layers, 0, floor)) {
        fprintf(stderr, "Floor FAILED: layer 0 should be skippable\n");
        return 1;
    }

    printf("ALL LAYER-SKIP TESTS PASSED (skip+run+floor)\n");
    return 0;
}
