/*
 * test_sparse_attn.c -- L11/L12 verification.
 */
#include "wubu_sparse_attn.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_sparse_attn (L11/L12) ===\n");

    /* 4x4 block scores; row 0 strongly prefers block 1 and 3. */
    float s[16];
    for (int i = 0; i < 16; i++) s[i] = 0.0f;
    s[0*4+1] = 0.9f; s[0*4+3] = 0.8f;
    s[1*4+0] = 0.7f; s[1*4+2] = 0.6f;
    uint8_t mask[16];
    int rc = wubu_block_sparse_mask(s, 4, 2, mask);
    CHECK(rc == 0, "mask returns 0");
    /* row 0 keeps blocks 1 and 3 (top-2) */
    CHECK(mask[0*4+1] == 1 && mask[0*4+3] == 1, "row0 keeps top-2 (1,3)");
    CHECK(mask[0*4+0] == 0 && mask[0*4+2] == 0, "row0 drops others");
    /* row 1 keeps 0 and 2 */
    CHECK(mask[1*4+0] == 1 && mask[1*4+2] == 1, "row1 keeps top-2 (0,2)");
    /* each row has exactly k kept */
    int cnt0 = 0; for (int i=0;i<4;i++) cnt0 += mask[0*4+i];
    CHECK(cnt0 == 2, "row0 keeps exactly k=2");

    CHECK(wubu_block_sparse_mask(NULL, 4, 2, mask) == -1, "null -> -1");
    CHECK(wubu_block_sparse_mask(s, 4, 0, mask) == -1, "k<=0 -> -1");

    /* L12 MoBA: 2 queries, 4 segments, top-1 each. */
    float seg[8];
    for (int i=0;i<8;i++) seg[i]=0.0f;
    seg[0*4+2]=0.9f; seg[1*4+1]=0.8f;
    uint8_t flags[8];
    int rc2 = wubu_moba_topk(seg, 2, 4, 1, flags);
    CHECK(rc2 == 0, "moba returns 0");
    CHECK(flags[0*4+2]==1 && flags[1*4+1]==1, "each query keeps top-1 segment");

    if (failures == 0) { printf("ALL SPARSE-ATTN TESTS PASSED\n"); return 0; }
    printf("%d SPARSE-ATTN TEST(S) FAILED\n", failures);
    return 1;
}
