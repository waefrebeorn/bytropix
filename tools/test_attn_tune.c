/*
 * test_attn_tune.c -- L06/N19/O11 verification.
 */
#include "wubu_attn_tune.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_attn_tune (L06/N19/O11) ===\n");

    /* L06 Quest top-k: pick top-2 of 6 block scores. */
    float s[6] = {0.1f, 0.9f, 0.2f, 0.8f, 0.3f, 0.05f};
    int out[6];
    int k = wubu_quest_topk(s, 6, 2, out);
    CHECK(k == 2, "select 2");
    int has1 = 0, has3 = 0;
    for (int i=0;i<k;i++) { if(out[i]==1) has1=1; if(out[i]==3) has3=1; }
    CHECK(has1 && has3, "top-2 are blocks 1,3 (scores 0.9,0.8)");
    CHECK(wubu_quest_topk(s, 6, 0, out) == 0, "k<=0 -> 0");
    CHECK(wubu_quest_topk(NULL, 6, 2, out) == 0, "null -> 0");

    /* N19 adaptive chunk: bigger workload -> bigger chunk (clamped). */
    int c_small = wubu_adaptive_chunk(64, 1, 8, 4096);
    int c_big   = wubu_adaptive_chunk(65536, 64, 8, 4096);
    CHECK(c_big >= c_small, "bigger workload -> >= chunk");
    CHECK(c_small >= 8 && c_big <= 4096, "chunk within [min,max]");
    CHECK(wubu_adaptive_chunk(0, 1, 8, 4096) == 8, "seq 0 -> min_c");

    /* O11 split-K: tokens=100, target=400 tiles -> k=4. */
    int sk = wubu_splitk_tune(100, 400, 16);
    CHECK(sk == 4, "split-K = ceil(400/100)=4");
    CHECK(wubu_splitk_tune(100, 400, 2) == 2, "split-K clamped to Kmax");

    if (failures == 0) { printf("ALL ATTN-TUNE TESTS PASSED\n"); return 0; }
    printf("%d ATTN-TUNE TEST(S) FAILED\n", failures);
    return 1;
}
