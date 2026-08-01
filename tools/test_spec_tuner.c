/*
 * test_spec_tuner.c -- N15/M12/N16 verification.
 */
#include "wubu_spec_tuner.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_spec_tuner (N15/M12/N16) ===\n");

    /* N15/M12: high acceptance -> large K; low -> small K. */
    wubu_spec_tuner_t *t = wubu_spec_tuner_create(4, 8, 0.5f);
    CHECK(t != NULL, "tuner create");
    for (int i = 0; i < 20; i++) wubu_spec_tuner_observe(t, 0, 9, 10); /* 0.9 acc */
    for (int i = 0; i < 20; i++) wubu_spec_tuner_observe(t, 1, 1, 10); /* 0.1 acc */
    int Khi = wubu_spec_tuner_K(t, 0);
    int Klo = wubu_spec_tuner_K(t, 1);
    CHECK(Khi > Klo, "high-acceptance layer -> larger K");
    CHECK(Khi <= 8 && Khi >= 1, "K within [1,Kmax]");
    CHECK(wubu_spec_tuner_K(t, -1) == 1, "OOB layer -> K=1");

    /* N16 cache feedback: warm prefix -> hit rate high. */
    wubu_cache_fb_t *c = wubu_cache_fb_create(0.2f);
    CHECK(c != NULL, "cache_fb create");
    for (int i = 0; i < 30; i++) wubu_cache_fb_observe(c, 9, 10); /* 0.9 */
    CHECK(fabs(wubu_cache_fb_hitrate(c) - 0.9) < 0.08, "hit rate ~0.9 after warm");
    wubu_cache_fb_observe(c, 0, 0); /* invalid ignored */
    wubu_cache_fb_destroy(c);
    wubu_spec_tuner_destroy(t);

    if (failures == 0) { printf("ALL SPEC-TUNER TESTS PASSED\n"); return 0; }
    printf("%d SPEC-TUNER TEST(S) FAILED\n", failures);
    return 1;
}
