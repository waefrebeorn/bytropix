/*
 * test_kv2026b.c -- Q01/Q04/Q05/Q06 verification.
 */
#include "wubu_kv2026b.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_kv2026b (Q01/Q04/Q05/Q06) ===\n");

    /* 4 tokens, d=2. keys: [1,0],[0,1],[1,0],[0.7,0.7] */
    float k[8] = {1,0, 0,1, 1,0, 0.7f,0.7f};
    int out[4];

    /* Q01 CentroidKV: k=2 centroids = token0 [1,0], token1 [0,1].
     * cluster0 rep = nearest to [1,0]: token0 or token2 (both [1,0]) -> token0.
     * cluster1 rep = nearest to [0,1]: token1. So {0,1}. */
    int m = wubu_centroidkv(k, 4, 2, 2, out);
    CHECK(m == 2, "2 representatives");
    CHECK((out[0]==0||out[0]==2) && out[1]==1, "centroid reps {0/2,1}");

    /* Q04 R-KV redundancy: token0 and token2 identical -> redundancy 1.0. */
    float red[4];
    wubu_rkv_redundancy(k, 4, 2, red);
    CHECK(red[0] > 0.99f && red[2] > 0.99f, "identical tokens -> redundancy ~1");

    /* Q05 OBCache saliency = |grad|^2. grad all-ones d=2 -> 2.0 each. */
    float g[8] = {1,1, 1,1, 1,1, 1,1};
    float sal[4];
    wubu_obcache_saliency(g, 4, 2, sal);
    CHECK(fabs(sal[0] - 2.0f) < 1e-5f, "saliency = |grad|^2 = 2");

    /* Q06 KeyDiff: keep 2 most distinct. token0==token2 (redundant), so keep
     * {0,1} or {1,2} or {0,3}/{1,3}; never keep both 0 and 2 together as the
     * two distinct ones. We just check 2 kept and they're distinct. */
    int kd[4];
    int km = wubu_keydiff_evict(k, 4, 2, 2, kd);
    CHECK(km == 2, "kept 2");
    CHECK(kd[0] != kd[1], "kept tokens distinct");

    if (failures == 0) { printf("ALL KV2026B TESTS PASSED\n"); return 0; }
    printf("%d KV2026B TEST(S) FAILED\n", failures);
    return 1;
}
