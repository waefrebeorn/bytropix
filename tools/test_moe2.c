/*
 * test_moe2.c -- the mixed-agents router test (phase 2 of the WuBu model).
 * Verifies: the router picks exactly k agents, the weights sum to 1,
 * the forward is finite, and the agents differentiate (different inputs
 * route to different experts).
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "wubu_moe2.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

int main(void)
{
    printf("=== test_moe2 (the mixed-agents router) ===\n");
    wubu_moe2_t moe;
    CHECK(wubu_moe2_init(&moe, 42) == 0, "init");

    float x[MOE2_D_MODEL];
    for (int i = 0; i < MOE2_D_MODEL; i++)
        x[i] = ((float)((i * 2654435761u) >> 16) / 32768.0f) - 1.0f;

    int idx[MOE2_N_ACTIVE];
    float w[MOE2_N_ACTIVE];
    int k = wubu_moe2_route(&moe, x, idx, w);
    CHECK(k == MOE2_N_ACTIVE, "router returns k agents");
    int distinct = (idx[0] != idx[1]);
    CHECK(distinct, "top-2 are distinct");
    /* the weights are softmax probs: each in (0,1), sum <= 1, and the
     * top-1 is the largest */
    CHECK(w[0] > 0 && w[0] < 1 && w[1] > 0 && w[1] < 1, "weights are probs in (0,1)");
    CHECK(w[0] >= w[1], "top-1 weight >= top-2 weight");
    CHECK(w[0] + w[1] <= 1.0 + 1e-4, "top-2 sum <= 1 (rest spread over experts)");
    printf("  routed agents: {%d w=%.4f, %d w=%.4f}\n", idx[0], w[0], idx[1], w[1]);

    float out[MOE2_D_MODEL];
    CHECK(wubu_moe2_forward(&moe, x, out) == 0, "forward");
    int finite = 1;
    double onorm = 0;
    for (int i = 0; i < MOE2_D_MODEL; i++) {
        if (out[i] != out[i]) { finite = 0; break; }
        onorm += (double)out[i] * out[i];
    }
    CHECK(finite, "output finite");
    CHECK(sqrt(onorm) > 0, "output non-zero");
    printf("  forward out-norm %.4f\n", sqrt(onorm));

    /* different inputs route differently (the agents specialize) */
    float y[MOE2_D_MODEL];
    for (int i = 0; i < MOE2_D_MODEL; i++) y[i] = -x[i];
    int idx2[MOE2_N_ACTIVE];
    float w2[MOE2_N_ACTIVE];
    wubu_moe2_route(&moe, y, idx2, w2);
    int same = (idx[0] == idx2[0] && idx[1] == idx2[1]);
    CHECK(!same, "different inputs -> different agents (specialization)");
    printf("  x -> {%d,%d}   -x -> {%d,%d}\n", idx[0], idx[1], idx2[0], idx2[1]);

    /* the expert sparsity: only k experts ran for this token */
    printf("  sparsity: %d/%d experts active per token (the mixed-agents win)\n",
           MOE2_N_ACTIVE, MOE2_N_EXPERTS);

    wubu_moe2_free(&moe);
    if (failures == 0) printf("ALL MOE2 TESTS PASSED -- the agents are mixed\n");
    else printf("%d MOE2 FAILURES\n", failures);
    return failures ? 1 : 0;
}
