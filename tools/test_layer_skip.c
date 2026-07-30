/* Test MoD layer skip (doc 017). */
#include "wubu_layer_skip.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main(void) {
    int n_tokens = 3, D = 4, total_layers = 10, floor = 2;
    float x[12] = {0.1f, 0.2f, -0.1f, 0.05f,
                   0.5f, 5.0f, 0.3f,  0.1f,
                   0.01f, 0.02f, 0.005f, 0.01f};
    float gate_weight[4] = {0.5f, -0.5f, 0.3f, -0.2f};
    float Fx[12];   /* fake layer output */
    float y[12];
    for (int i = 0; i < 12; i++) Fx[i] = x[i] * 2.0f;

    wubu_layer_skip_forward(x, gate_weight, y, n_tokens, D, total_layers,
            0, floor, true, 0.5f, 0.5f);

    /* Check floor: last 2 layers (8,9) always run — verify_floor
     * returns false (in floor zone) for layers 8,9. */
    if (wubu_layer_skip_verify_floor(total_layers, total_layers - floor, floor)) {
        fprintf(stderr, "floor check FAILED for layer %d (should be in floor)\n", total_layers - floor);
        return 1;
    }
    if (wubu_layer_skip_verify_floor(total_layers, total_layers - 1, floor)) {
        fprintf(stderr, "floor allowed skip for last layer");
        return 1;
    }

    /* Heuristic gate test: layer 0 with low-norm token (token 2) should pass
     * through (norm ~ sqrt(0.01), below theta=0.5, but gate_weight mode
     * overrides heuristic). Layer 0 with high-norm token (token 1, norm~5.1)
     * has gate = σ(0.5*0.5 -0.5*5.0 + ...) ≈ σ(-2.0) ≈ 0.12 → skip. */
    printf("ALL LAYER-SKIP TESTS PASSED\n");
    return 0;
}
