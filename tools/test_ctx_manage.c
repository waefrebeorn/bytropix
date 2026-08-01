/*
 * test_ctx_manage.c -- L16/N07/N14 verification.
 */
#include "wubu_ctx_manage.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_ctx_manage (L16/N07/N14) ===\n");

    /* L16 elastic: peaky (e=0) -> shrink toward wmin; diffuse (e=1) -> grow. */
    int shrink = wubu_elastic_window(1000, 0.0f, 256, 4096, 1.0f);
    int grow   = wubu_elastic_window(1000, 1.0f, 256, 4096, 1.0f);
    CHECK(shrink == 256, "peaky -> wmin");
    CHECK(grow == 4096, "diffuse -> wmax");
    CHECK(wubu_elastic_window(1000, 0.5f, 256, 4096, 1.0f) > 256 &&
          wubu_elastic_window(1000, 0.5f, 256, 4096, 1.0f) < 4096, "mid -> between");

    /* N07 tiered: hot+heavy -> HOT(0); cold -> COLD(2). */
    CHECK(wubu_tier_advice(0.9f, 0.9f) == 0, "hot+heavy -> HOT");
    CHECK(wubu_tier_advice(0.1f, 0.05f) == 2, "cold -> COLD");
    CHECK(wubu_tier_advice(0.5f, 0.5f) == 1, "mid -> WARM");

    /* N14 MoD: skipping too little (measured<target) -> raise tau. */
    float t0 = wubu_mod_tau(0.3f, 0.7f, 0.2f, 0.5f);
    CHECK(t0 > 0.3f, "raise tau when under-skipping");
    float t1 = wubu_mod_tau(0.8f, 0.3f, 0.9f, 0.5f);
    CHECK(t1 < 0.8f, "lower tau when over-skipping");
    CHECK(t0 >= 0.0f && t0 <= 1.0f, "tau in [0,1]");

    if (failures == 0) { printf("ALL CTX-MANAGE TESTS PASSED\n"); return 0; }
    printf("%d CTX-MANAGE TEST(S) FAILED\n", failures);
    return 1;
}
