/*
 * test_kv2026c.c -- Q11/Q19/R04/R05 verification.
 */
#include "wubu_kv2026c.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_kv2026c (Q11/Q19/R04/R05) ===\n");

    /* Q11 DASH-KV schedule: 6 tokens d=2, 2 buckets, scores favor bucket of
     * tokens 0,2,4. We just check a valid winning bucket in [0,2) and that
     * out_bucket is filled with valid bucket ids. */
    float k[12] = {1,0, 2,0, 1,1, 3,1, 0,2, 1,2};
    float sc[6] = {5,1, 5,1, 5,1};
    int ob[6];
    int win = wubu_dashkv_schedule(k, 6, 2, sc, 2, ob);
    CHECK(win == 0 || win == 1, "winning bucket in range");
    int ok = 1; for (int i=0;i<6;i++) if (ob[i]<0||ob[i]>=2) ok=0;
    CHECK(ok, "all tokens assigned a valid bucket");

    /* Q19 HeteroCache: sharp head (e=0) -> bmax; diffuse (e=1) -> bmin. */
    float ent[2] = {0.0f, 1.0f};
    int bits[2];
    wubu_hetero_bits(ent, 2, 2, 8, bits);
    CHECK(bits[0] == 8 && bits[1] == 2, "sharp->bmax, diffuse->bmin");

    /* R04 redundancy profile: tokens 0,2 reasoning with redundancy 1.0,1.0;
     * token1 not reasoning. Mean = 1.0. */
    float red[3] = {1.0f, 0.0f, 1.0f};
    char reason[3] = {1, 0, 1};
    float p = wubu_redundancy_profile(red, reason, 3);
    CHECK(fabs(p - 1.0f) < 1e-5f, "reasoning redundancy mean = 1.0");

    /* R05 multi-agent coherence: 2 identical agents -> 1.0. */
    float sums[4] = {1,0, 1,0};
    CHECK(fabs(wubu_multiagent_coherence(sums, 2, 2) - 1.0f) < 1e-5f, "identical -> 1.0");

    if (failures == 0) { printf("ALL KV2026C TESTS PASSED\n"); return 0; }
    printf("%d KV2026C TEST(S) FAILED\n", failures);
    return 1;
}
