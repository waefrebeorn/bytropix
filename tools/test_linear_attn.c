/*
 * test_linear_attn.c -- S01/S03/S04/S05/S07 verification.
 */
#include "wubu_linear_attn.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_linear_attn (S01/S03/S04/S05/S07) ===\n");
    int d = 2;
    float S[4] = {0,0, 0,0};       /* zero initial state */
    float k[2] = {1,0};
    float v[2] = {0,1};
    float S2[4];

    /* S01 DeltaNet from zero: S' = -beta*(-v) k^T = beta*v k^T = beta*[0,1][1,0]
     * = beta*[0,0; 1,0]. With beta=1 -> [0,0;1,0]. */
    wubu_deltanet_update(S, k, v, d, 1.0f, S2);
    CHECK(fabsf(S2[0]-0)<1e-5f && fabsf(S2[1]-0)<1e-5f &&
          fabsf(S2[2]-1)<1e-5f && fabsf(S2[3]-0)<1e-5f, "DeltaNet S'=[0,0;1,0]");

    /* S03 Mamba-2 from zero: S' = b*k v^T. k=[1,0],v=[0,1] -> [[0,1],[0,0]]. */
    wubu_mamba2_update(S, k, v, d, 0.0f, 1.0f, S2);
    CHECK(fabsf(S2[1]-1)<1e-5f, "Mamba2 S'=[0,1;0,0] (S2[1]=1)");

    /* S04 GLA from zero: S' = k v^T. */
    wubu_gla_update(S, k, v, d, 1.0f, S2);
    CHECK(fabsf(S2[1]-1)<1e-5f, "GLA S' has [1]=1");

    /* S05 RetNet from zero. */
    wubu_retnet_update(S, k, v, d, 1.0f, S2);
    CHECK(fabsf(S2[1]-1)<1e-5f, "RetNet S' has [1]=1");

    /* S07 HGRN2 from zero: S' = g*k v^T. */
    wubu_hgrn2_update(S, k, v, d, 1.0f, S2);
    CHECK(fabsf(S2[1]-1)<1e-5f, "HGRN2 S' has [1]=1");

    if (failures == 0) { printf("ALL LINEAR-ATTN TESTS PASSED\n"); return 0; }
    printf("%d LINEAR-ATTN TEST(S) FAILED\n", failures);
    return 1;
}
