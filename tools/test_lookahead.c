/*
 * test_lookahead.c -- M06 verification.
 */
#include "wubu_lookahead.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_lookahead (M06) ===\n");

    /* history: ... 5 7 5 7 9 -> at pos 4, suffix [5,7] (idx 2,3) earlier occurs
     * at idx 0,1; token after that earlier occurrence is history[2] = 5. */
    int h[5] = {5, 7, 5, 7, 9};
    int p = wubu_lookahead_probe(h, 5, 4, 2); /* pos=4: suffix [5,7] at end */
    CHECK(p == 5, "n-gram [5,7] -> next 5 (after earlier occurrence)");
    /* pos=2: suffix [7] (n=1) -> no strictly-earlier occurrence -> -1 */
    int p2 = wubu_lookahead_probe(h, 5, 2, 1);
    CHECK(p2 == -1, "unigram 7 has no earlier occurrence -> -1");
    /* longer history: [1,2,3,1,2,3,4] at pos=6, n=3, suffix [1,2,3] at idx 3,4,5;
     * earliest prior occurrence at idx 0,1,2; token after it is history[3] = 1. */
    int h2[7] = {1, 2, 3, 1, 2, 3, 4};
    int p3 = wubu_lookahead_probe(h2, 7, 6, 3);
    CHECK(p3 == 1, "n-gram [1,2,3] -> next 1 (after earliest prior occurrence)");

    CHECK(wubu_lookahead_probe(h, 5, 1, 2) == -1, "too short -> -1");
    CHECK(wubu_lookahead_probe(NULL, 5, 4, 2) == -1, "null -> -1");
    CHECK(wubu_lookahead_probe(h, 5, 4, 0) == -1, "n<=0 -> -1");

    if (failures == 0) { printf("ALL LOOKAHEAD TESTS PASSED\n"); return 0; }
    printf("%d LOOKAHEAD TEST(S) FAILED\n", failures);
    return 1;
}
