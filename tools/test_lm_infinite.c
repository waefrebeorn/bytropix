/*
 * test_lm_infinite.c -- L13/O07/N20 verification.
 */
#include "wubu_lm_infinite.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_lm_infinite (L13/O07/N20) ===\n");

    /* L13 landmarks every 4 in seq 12 -> positions 4,8. */
    int lm[8];
    int nl = wubu_landmark_positions(12, 4, lm);
    CHECK(nl == 2 && lm[0] == 4 && lm[1] == 8, "landmarks at 4,8");
    CHECK(wubu_landmark_positions(10, 0, lm) == 0, "stride 0 -> 0");

    /* O07 sinks: first 2 of 6 -> positions 0,1. */
    int sk[8];
    int ns = wubu_sink_positions(6, 2, sk);
    CHECK(ns == 2 && sk[0] == 0 && sk[1] == 1, "sinks at 0,1");
    CHECK(wubu_sink_positions(6, 99, sk) == 6, "n_sink clamped to seq");

    /* N20 shadow quant: warm=3, switch to cheap after 3 matches. */
    wubu_shadow_t *s = wubu_shadow_create(16, 2, 3);
    CHECK(wubu_shadow_observe(s, 1) == 16, "still ref after 1 match");
    wubu_shadow_observe(s, 1);
    CHECK(wubu_shadow_observe(s, 1) == 2, "switches to cheap after 3 matches");
    CHECK(wubu_shadow_observe(s, 0) == 2, "stays cheap even if later diverge");
    wubu_shadow_destroy(s);
    CHECK(wubu_shadow_create(0, 2, 3) == NULL, "bad bits -> NULL");

    if (failures == 0) { printf("ALL LM-INFINITE TESTS PASSED\n"); return 0; }
    printf("%d LM-INFINITE TEST(S) FAILED\n", failures);
    return 1;
}
