/* Test AWQ activation-aware weight quantization (doc B05). */
#include "wubu_awq.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

int main(void) {
    int n = 100;
    float act_mag[100], weight_mag[100];
    for (int i = 0; i < 100; i++) {
        act_mag[i] = (i == 7 || i == 42 || i == 99) ? 10.0f : 0.1f;
        weight_mag[i] = 1.0f;
    }

    /* Test 1: find salient — top 1% = 1 channel out of 100 */
    bool salient[100];
    wubu_awq_find_salient(act_mag, 100, 0.01f, salient);
    int n_salient = 0;
    for (int i = 0; i < 100; i++) if (salient[i]) n_salient++;
    printf("Salient channels: %d (expected ~1)\n", n_salient);
    assert(n_salient >= 1 && n_salient <= 3);

    /* Test 2: compute scales — salient should get scale > 1 */
    float scales[100];
    wubu_awq_compute_scales(act_mag, weight_mag, 100, 0.5f, scales);
    for (int i = 0; i < 100; i++) {
        if (salient[i]) {
            printf("Salient channel %d: scale=%.4f (should be > 1)\n", i, (double)scales[i]);
            assert(scales[i] > 1.0f);
        }
    }

    /* Test 3: matmul identity — (W*s) * (x/s) = W*x */
    float W[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float s[4] = {2.0f, 2.0f, 1.0f, 1.0f};  /* scale first 2 channels */

    float ref = 0.0f;
    for (int i = 0; i < 4; i++) ref += W[i] * x[i];

    wubu_awq_apply_scale_weights(W, s, 4);
    wubu_awq_apply_scale_activations(x, s, 4);
    float scaled = 0.0f;
    for (int i = 0; i < 4; i++) scaled += W[i] * x[i];

    printf("Original matmul: %.4f\n", (double)ref);
    printf("Scaled matmul:   %.4f (should match)\n", (double)scaled);
    assert(fabsf(scaled - ref) < 1e-4f);

    printf("ALL AWQ TESTS PASSED\n");
    return 0;
}
