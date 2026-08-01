/*
 * test_kv_evict_h2o.c -- L03 H2O heavy-hitter eviction verification.
 * Verifies: cumulative attention mass drives retention, top-p% kept,
 * edge cases (keep_frac=1 keep all, n=0, attn<0 ignored).
 */
#include "wubu_kv_evict.h"
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_kv_evict_h2o (L03 H2O heavy-hitter) ===\n");

    /* Feed 5 blocks; block 2 and 4 get high attention (heavy hitters). */
    wubu_kv_evict_t *e = wubu_kv_evict_create(0.95f);
    wubu_kv_evict_set_h2o(e, 1);
    for (int b = 0; b < 5; b++) wubu_kv_evict_track(e, b, 0.0f);
    wubu_kv_evict_track_attn(e, 0, 0.10f);
    wubu_kv_evict_track_attn(e, 1, 0.10f);
    wubu_kv_evict_track_attn(e, 2, 0.50f);   /* heavy hitter */
    wubu_kv_evict_track_attn(e, 3, 0.10f);
    wubu_kv_evict_track_attn(e, 4, 0.40f);   /* heavy hitter */
    wubu_kv_evict_track_attn(e, 2, -5.0f);   /* negative ignored */

    CHECK(wubu_kv_evict_count(e) == 5, "tracking 5 blocks");

    /* keep top 40% (2 of 5) -> evict 3 lowest-attention (0,1,3) */
    int victims[8];
    int n = wubu_kv_evict_select_h2o(e, victims, 8, 0.4f);
    CHECK(n == 3, "evict 3 of 5 at keep_frac=0.4");
    int has0 = 0, has1 = 0, has3 = 0, has2 = 0, has4 = 0;
    for (int i = 0; i < n; i++) {
        if (victims[i] == 0) has0 = 1;
        if (victims[i] == 1) has1 = 1;
        if (victims[i] == 3) has3 = 1;
        if (victims[i] == 2) has2 = 1;
        if (victims[i] == 4) has4 = 1;
    }
    CHECK(has0 && has1 && has3, "low-attention blocks (0,1,3) evicted");
    CHECK(!has2 && !has4, "heavy hitters (2,4) retained");

    /* keep_frac=1.0 => keep all (no eviction) */
    int v2[8];
    CHECK(wubu_kv_evict_select_h2o(e, v2, 8, 1.0f) == 0, "keep_frac=1 keeps all");
    CHECK(wubu_kv_evict_select_h2o(e, v2, 8, 2.0f) == 0, "keep_frac>1 keeps all");

    /* empty tracker */
    wubu_kv_evict_t *e2 = wubu_kv_evict_create(0.95f);
    CHECK(wubu_kv_evict_select_h2o(e2, v2, 8, 0.5f) == 0, "empty -> 0 victims");
    wubu_kv_evict_free(e2);

    wubu_kv_evict_free(e);
    if (failures == 0) { printf("ALL H2O EVICT TESTS PASSED\n"); return 0; }
    printf("%d H2O EVICT TEST(S) FAILED\n", failures);
    return 1;
}
