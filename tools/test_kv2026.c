/*
 * test_kv2026.c -- Q02/Q03/Q07/Q09/Q10 verification.
 */
#include "wubu_kv2026.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_kv2026 (Q02/Q03/Q07/Q09/Q10) ===\n");

    /* Q02 ChunkKV: 8 tokens, 4 chunks of 2, keep 2 highest-mean chunks.
     * scores [1,1, 9,9, 1,1, 9,9] -> chunks means [1,9,1,9]; keep chunks 1,3
     * -> tokens {2,3,6,7}. */
    float s[8] = {1,1, 9,9, 1,1, 9,9};
    int out[8];
    int m = wubu_chunkkv_evict(s, 8, 4, 2, out);
    CHECK(m == 4, "kept 4 tokens (2 chunks)");
    int has2=0,has3=0,has6=0,has7=0;
    for (int i=0;i<m;i++){ if(out[i]==2)has2=1; if(out[i]==3)has3=1; if(out[i]==6)has6=1; if(out[i]==7)has7=1; }
    CHECK(has2&&has3&&has6&&has7, "kept highest-mean chunks {2,3,6,7}");

    /* Q03 KVzip importance: token with varying attention across heads -> high var.
     * 2 heads, token0 gets [0,1] (var .25), token1 gets [0.5,0.5] (var 0). */
    float attn[4] = {0.0f, 1.0f, 0.5f, 0.5f}; /* [h0t0,h1t0, h0t1,h1t1] */
    float imp[2];
    wubu_kvzip_importance(attn, 2, 2, imp);
    CHECK(imp[0] > imp[1], "varying-attn token has higher reconstruction value");

    /* Q07 LAVa: sharp (e=0) -> cap; diffuse (e=1) -> 1. */
    CHECK(wubu_lava_budget(0.0f, 0.0f, 8) == 8, "sharpest -> cap");
    CHECK(wubu_lava_budget(1.0f, 1.0f, 8) == 1, "diffuse -> min 1");

    /* Q09 FreeKV top-k: scores [3,1,4,2], k=2 -> {2(idx2=4),0(idx0=3)}. */
    float fs[4] = {3,1,4,2};
    int fk[4];
    wubu_freekv_topk(fs, 4, 2, fk);
    CHECK(fk[0]==2 && fk[1]==0, "top-2 = {2,0}");

    /* Q10 TTKV tiers. */
    CHECK(wubu_ttkv_tier(1, 10, 100) == 0, "age<warm -> HOT");
    CHECK(wubu_ttkv_tier(50, 10, 100) == 1, "warm<=age<cold -> WARM");
    CHECK(wubu_ttkv_tier(200, 10, 100) == 2, "age>=cold -> COLD");

    if (failures == 0) { printf("ALL KV2026 TESTS PASSED\n"); return 0; }
    printf("%d KV2026 TEST(S) FAILED\n", failures);
    return 1;
}
