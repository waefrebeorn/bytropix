/*
 * test_dn2.c -- S02/T04 verification.
 */
#include "wubu_dn2.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_dn2 (S02/T04) ===\n");
    int d = 2;
    float S[4] = {0,0, 0,0};
    float k[2] = {1,0};
    float v[2] = {0,1};
    float S2[4];

    /* S02 GDN-2 from zero, k=[1,0],v=[0,1]: delta-rule stores assoc at S[2].
     * e=1,w=1 -> S'=[0,0;1,0] (S2[2]=1). */
    wubu_dn2_update(S, k, v, d, 1.0f, 1.0f, 1.0f, S2);
    CHECK(fabsf(S2[2]-1)<1e-5f, "GDN-2 e1w1 S'=[0,0;1,0] (S2[2]=1)");

    /* erase=0, write=0: no-op -> S'=0. */
    wubu_dn2_update(S, k, v, d, 1.0f, 0.0f, 0.0f, S2);
    CHECK(fabsf(S2[0])<1e-5f && fabsf(S2[1])<1e-5f &&
          fabsf(S2[2])<1e-5f && fabsf(S2[3])<1e-5f, "e0w0 -> S'=0 (no-op)");

    /* erase=1, write=0 from zero: erase step writes the single assoc -> S2[2]=1. */
    wubu_dn2_update(S, k, v, d, 1.0f, 1.0f, 0.0f, S2);
    CHECK(fabsf(S2[2]-1)<1e-5f, "e1w0 from zero -> S2[2]=1");

    /* T04 STE: x=0.5,thr=0.5 -> ternary (0.5 not >0.5) = 0; grad passes (|0.5|<=1). */
    float t; int gp;
    wubu_ternary_ste(0.5f, 0.5f, &t, &gp);
    CHECK(fabsf(t) < 1e-5f, "STE 0.5 -> 0");
    CHECK(gp == 1, "STE grad passes for |0.5|<=1");
    wubu_ternary_ste(2.0f, 0.5f, &t, &gp);
    CHECK(fabsf(t-1.0f)<1e-5f, "STE 2.0 -> 1");
    CHECK(gp == 0, "STE grad blocked for |2|>1");

    if (failures == 0) { printf("ALL DN2 TESTS PASSED\n"); return 0; }
    printf("%d DN2 TEST(S) FAILED\n", failures);
    return 1;
}
