/*
 * test_lruk.c -- O01 LRU-k verification.
 */
#include "wubu_lruk.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_lruk (O01 LRU-k) ===\n");

    wubu_lruk_t *e = wubu_lruk_create(4);
    CHECK(e != NULL, "lruk create");
    wubu_lruk_touch(e, 1);
    wubu_lruk_touch(e, 2);
    wubu_lruk_touch(e, 3);
    /* re-touch 1 so it becomes most-recently-used */
    wubu_lruk_touch(e, 1);
    CHECK(wubu_lruk_count(e) == 3, "tracked 3");
    /* evict k=1 -> least recently used is 2 (touched before 3, before re-touch 1) */
    int out[4];
    int k = wubu_lruk_select(e, 1, out);
    CHECK(k == 1 && out[0] == 2, "evict LRU (id 2)");
    /* evict k=2 -> two least recent are 2 then 3 (select is read-only survey) */
    int out2[4];
    int k2 = wubu_lruk_select(e, 2, out2);
    CHECK(k2 == 2 && out2[0] == 2 && out2[1] == 3, "evict two LRU (2,3)");

    CHECK(wubu_lruk_create(0) == NULL, "cap 0 -> NULL");
    CHECK(wubu_lruk_select(e, 5, out) <= 3, "k clamped to n");

    wubu_lruk_destroy(e);
    if (failures == 0) { printf("ALL LRUK TESTS PASSED\n"); return 0; }
    printf("%d LRUK TEST(S) FAILED\n", failures);
    return 1;
}
