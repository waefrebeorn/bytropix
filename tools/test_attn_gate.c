/* Test: wubu_attn_gate (doc 011) — gated attention sink-free forward. */
#include "wubu_attn_gate.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main(void) {
    int D = 8, H = 4;
    float x[4] = {1.0f, 0.5f, -0.3f, 0.2f};
    float W_gate[8 * 4];
    for (int i = 0; i < D * H; i++) W_gate[i] = 0.1f * ((i % 7) - 3);
    float attn_out[8];
    for (int i = 0; i < D; i++) attn_out[i] = (float)(i + 1);
    float y[8];

    /* Reference: per-d sigmoid(W_gate * x) */
    float ref[8];
    for (int d = 0; d < D; d++) {
        float dot = 0.0f;
        for (int k = 0; k < H; k++) dot += W_gate[d * H + k] * x[k];
        ref[d] = attn_out[d] / (1.0f + expf(-dot));
    }

    wubu_attn_gate_forward(attn_out, x, W_gate, y, D, H);
    float max_err = 0.0f;
    for (int d = 0; d < D; d++) {
        float e = fabsf(y[d] - ref[d]);
        if (e > max_err) max_err = e;
    }
    if (max_err > 1e-5f) {
        fprintf(stderr, "max_err=%g too large\n", (double)max_err);
        return 1;
    }
    printf("ALL ATTN-GATE TESTS PASSED (max_err=%.2e)\n", (double)max_err);
    return 0;
}
