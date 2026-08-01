/*
 * test_misc_gaps.c -- L05/O12/O13/O15/P12/P13 verification.
 */
#include "wubu_misc_gaps.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_misc_gaps (L05/O12/O13/O15/P12/P13) ===\n");

    /* L05 CacheBlend LCP: [1,2,3,4] vs [1,2,9,9] -> 2. */
    int a[4] = {1,2,3,4}, b[4] = {1,2,9,9};
    CHECK(wubu_lcp_len(a, b, 4) == 2, "LCP length 2");
    CHECK(wubu_lcp_len(a, b, 0) == 0, "n=0 -> 0");

    /* O12 dequant equivalence: round-trip within tol. */
    float x[3] = {0.5f, -0.25f, 1.0f};
    CHECK(wubu_dequant_equiv(x, 3, 100.0f, 0.02f) == 1, "quant round-trip within tol");
    CHECK(wubu_dequant_equiv(x, 3, 2.0f, 0.01f) == 0, "coarse quant exceeds tol");

    /* O13 prefault: best-effort, returns int (non-fatal). Call on a small buffer. */
    float buf[16];
    wubu_prefault(buf, sizeof(buf));  /* should not crash */
    CHECK(1, "prefault did not crash");

    /* O15 rhythmic gate in [0,1]. */
    float g0 = wubu_rhythmic_gate(0, 0.01f, 0.0f);
    CHECK(g0 >= 0.0f && g0 <= 1.0f, "gate in [0,1]");

    /* P12 prefetch: best-effort, no crash. */
    wubu_kv_prefetch(buf, 16, 4);  /* should not crash */
    CHECK(1, "prefetch did not crash");

    /* P13 fused RoPE+quant: rotate (1,0) by 90deg -> (0,1), quant to 8 bits. */
    unsigned char out[2] = {0};
    wubu_fused_rope_quant(1.0f, 0.0f, 1.5707963f, 8, 1.0f, out);
    CHECK(out[0] == 128 && out[1] == 255, "RoPE+quant (1,0)->90deg -> (128,255)");

    if (failures == 0) { printf("ALL MISC-GAPS TESTS PASSED\n"); return 0; }
    printf("%d MISC-GAPS TEST(S) FAILED\n", failures);
    return 1;
}
