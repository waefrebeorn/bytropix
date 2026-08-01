/*
 * test_ternary.c -- T01/T02/T03 verification.
 */
#include "wubu_ternary.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_ternary (T01/T02/T03) ===\n");

    /* T03 scale: max|w| = 2.0 */
    float w[4] = {1.0f, -2.0f, 0.5f, 0.0f};
    float scale = wubu_ternary_scale(w, 4);
    CHECK(fabsf(scale - 2.0f) < 1e-5f, "absmax scale = 2.0");

    /* T01 pack then T03-unpack: round-trip (ternary is lossy by design, but
     * dequant recovers the ternary values). */
    unsigned char packed[1];
    int nb = wubu_ternary_pack(w, 4, scale, 0.4f, packed);
    CHECK(nb == 1, "4 values -> 1 byte");
    float back[4];
    wubu_ternary_unpack(packed, 4, scale, back);
    /* values beyond 0.5*scale=1.0 stay; within -> 0. w=[1,-2,0.5,0] -> ternary
     * [1,-1,0,0] -> dequant [2,-2,0,0]. */
    CHECK(fabsf(back[0]-2.0f)<1e-5f && fabsf(back[1]+2.0f)<1e-5f &&
          fabsf(back[2])<1e-5f && fabsf(back[3])<1e-5f, "round-trip ternary");

    /* T02 mpGEMV: 1 row, 4 cols, act=[1,1,1,1] -> y = scale*sum(ternary).
     * ternary row = [1,-1,0,0] -> sum=0 -> y=0. */
    float act[4] = {1,1,1,1};
    float y[1];
    int r = wubu_mpgemv(packed, 1, 4, scale, act, y);
    CHECK(r == 1 && fabsf(y[0]) < 1e-5f, "mpGEMV sum = 0");

    if (failures == 0) { printf("ALL TERNARY TESTS PASSED\n"); return 0; }
    printf("%d TERNARY TEST(S) FAILED\n", failures);
    return 1;
}
