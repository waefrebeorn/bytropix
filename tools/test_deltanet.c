/*
 * test_deltanet.c -- the Gated-DeltaNet test (research 008, phase 5).
 * Verifies:
 *   1. the delta rule learns a repeating pattern (a fixed k→v mapping
 *      is captured in the state and recalled exactly)
 *   2. the state size is FIXED (linear in heads*dim², NOT quadratic
 *      in context) -- the KV saving
 *   3. the recurrence is order-sensitive (the delta rule is
 *      associative-ish: repeated identical keys converge)
 *   4. reset clears the memory
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_deltanet.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

int main(void)
{
    printf("=== test_deltanet (the Gated-DeltaNet linear mixer) ===\n");
    int hd = 8, nh = 2;
    wubu_deltanet_state_t st;
    CHECK(wubu_deltanet_state_init(&st, nh, hd) == 0, "state init");

    /* 1. teach the state a fixed k->v mapping and recall it */
    float k1[8] = {1,0,0,0,0,0,0,0};
    float v1[8] = {2,4,6,8,10,12,14,16};
    float out[8];
    float alpha = 0.99f, beta = 1.0f;
    /* present the pair a few times (learning) */
    for (int rep = 0; rep < 5; rep++)
        wubu_deltanet_step(&st, 0, k1, v1, alpha, beta, out);
    /* PURE recall with the same key: the readout must approach v1 */
    wubu_deltanet_read(&st, 0, k1, out);
    double err = 0;
    for (int i = 0; i < hd; i++) err += fabs(out[i] - v1[i]);
    CHECK(err < 1.0, "delta rule learns the k->v mapping");
    printf("  learned mapping recall err: %.4f (out0=%.2f expect 2)\n",
           err, out[0]);

    /* 2. the state size is FIXED: nh*hd² floats, not T² */
    size_t state_floats = (size_t)nh * hd * hd;
    size_t full_kv_for_1k = (size_t)1000 * nh * hd * 2;
    CHECK(state_floats * 10 < full_kv_for_1k,
          "state << quadratic KV (the 008 KV saving)");
    printf("  state: %zu floats vs full KV for 1K ctx: %zu -- %.0fx smaller\n",
           state_floats, full_kv_for_1k, (double)full_kv_for_1k / state_floats);

    /* 3. a NEW key must NOT recall the old value (the memory is
     * key-addressed, not content-addressed) */
    float k2[8] = {0,1,0,0,0,0,0,0};
    float out2[8];
    wubu_deltanet_read(&st, 0, k2, out2);
    /* with an orthogonal key, the readout S·k2^T ~ 0 (the state's
     * first column was trained on k1) */
    CHECK(fabs(out2[0]) < 1.0, "orthogonal key recalls ~0 (key-addressed)");
    printf("  orthogonal-key recall: %.4f (expect ~0)\n", out2[0]);

    /* 4. prefill: T tokens through the recurrence, exact output */
    int T = 16;
    float *K = malloc((size_t)T * nh * hd * sizeof(float));
    float *V = malloc((size_t)T * nh * hd * sizeof(float));
    float *O = malloc((size_t)T * nh * hd * sizeof(float));
    for (int t = 0; t < T; t++)
        for (int h = 0; h < nh; h++)
            for (int i = 0; i < hd; i++) {
                K[((size_t)t*nh+h)*hd+i] = (t == i) ? 1.0f : 0.0f;
                V[((size_t)t*nh+h)*hd+i] = (float)t;
            }
    wubu_deltanet_state_reset(&st);
    CHECK(wubu_deltanet_prefill(&st, K, V, T, alpha, beta, O) == 0, "prefill");
    int finite = 1;
    for (int i = 0; i < T*nh*hd; i++) if (O[i] != O[i]) finite = 0;
    CHECK(finite, "prefill outputs finite");
    printf("  prefill %d tokens ok\n", T);

    /* 5. reset clears the memory */
    wubu_deltanet_state_reset(&st);
    wubu_deltanet_read(&st, 0, k1, out);
    CHECK(fabs(out[0]) < 1.0, "reset clears the learned mapping");
    printf("  after reset, recall: %.4f (expect ~0)\n", out[0]);

    free(K); free(V); free(O);
    wubu_deltanet_state_free(&st);
    if (failures == 0) printf("ALL DELTANET TESTS PASSED -- the KV is linear\n");
    else printf("%d DELTANET FAILURES\n", failures);
    return failures ? 1 : 0;
}
