/*
 * test_wm_kv.c -- O05/N02/N08 verification.
 */
#include "wubu_wm_kv.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_wm_kv (O05/N02/N08) ===\n");

    /* O05 bounded WM ring: capacity 4, push 6 -> evicts oldest two (0,1). */
    wubu_wm_kv_t *w = wubu_wm_kv_create(4);
    CHECK(w != NULL, "wm create");
    int ev;
    ev = wubu_wm_kv_push(w, 0); CHECK(ev == -1, "no evict at 1 (cap4)");
    wubu_wm_kv_push(w, 1);
    wubu_wm_kv_push(w, 2);
    wubu_wm_kv_push(w, 3);
    CHECK(wubu_wm_kv_count(w) == 4, "full at cap 4");
    ev = wubu_wm_kv_push(w, 4); CHECK(ev == 0, "evict oldest (0) at 5th");
    ev = wubu_wm_kv_push(w, 5); CHECK(ev == 1, "evict next oldest (1) at 6th");
    CHECK(wubu_wm_kv_count(w) == 4, "still bounded at 4 after overflow");
    wubu_wm_kv_destroy(w);
    CHECK(wubu_wm_kv_create(0) == NULL, "cap 0 -> NULL");

    /* N02 online roofline: EMA tracks observations. */
    wubu_roofline_t *r = wubu_roofline_create(1e9, 0.1);
    CHECK(r != NULL, "roofline create");
    wubu_roofline_observe(r, 1e9, 0.02);   /* 50 GB/s */
    wubu_roofline_observe(r, 1e9, 0.02);
    double b = wubu_roofline_beta(r);
    CHECK(fabs(b - 50e9) < 1e9, "beta settles ~50 GB/s");
    wubu_roofline_observe(r, 1e9, 0.05);   /* 20 GB/s -> EMA drifts down */
    CHECK(wubu_roofline_beta(r) < b, "beta drifts down after slow obs");
    wubu_roofline_observe(r, 0.0, 0.0);    /* invalid ignored */
    wubu_roofline_destroy(r);

    /* N08 per-layer floor: deeper >= shallower, within [min,1]. */
    float f0 = wubu_layer_floor(0, 32, 0.2f);
    float f31 = wubu_layer_floor(31, 32, 0.2f);
    CHECK(f0 <= f31, "deeper layer floor >= shallower");
    CHECK(f0 >= 0.2f && f31 <= 1.0f, "floor within [min,1]");
    CHECK(wubu_layer_floor(-1, 32, 0.2f) >= 0.2f, "OOB layer -> min_floor");

    if (failures == 0) { printf("ALL WM-KV TESTS PASSED\n"); return 0; }
    printf("%d WM-KV TEST(S) FAILED\n", failures);
    return 1;
}
