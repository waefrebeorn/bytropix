/*
 * test_agentic_kv.c -- S06/U01/U02/U03/U04/U05 verification.
 */
#include "wubu_agentic_kv.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_agentic_kv (S06/U01/U02/U03/U04/U05) ===\n");

    /* S06 hybrid 3:1 mix (period 4): layers 0,4,8 -> attention(0); others recurrent(1). */
    CHECK(wubu_hybrid_is_recurrent(0, 4) == 0, "layer 0 -> attention");
    CHECK(wubu_hybrid_is_recurrent(1, 4) == 1, "layer 1 -> recurrent");
    CHECK(wubu_hybrid_is_recurrent(4, 4) == 0, "layer 4 -> attention");

    /* U01 shared-KV: layer 6 with off 2 -> source layer 4. */
    CHECK(wubu_shared_kv_source(6, 2) == 4, "layer6 off2 -> src4");
    CHECK(wubu_shared_kv_source(1, 2) == 1, "layer1 < off -> self");

    /* U02 CSA: 4 tokens d=2, group 2 -> 2 compressed entries (mean-pool). */
    float k[8] = {0,0, 2,2, 4,4, 6,6};
    float co[8];
    int cc = wubu_csa_compress(k, 4, 2, 2, co);
    CHECK(cc == 2, "4 tokens -> 2 entries");
    CHECK(fabsf(co[0]-1.0f)<1e-5f && fabsf(co[1]-1.0f)<1e-5f, "entry0 mean = 1");
    CHECK(fabsf(co[2]-5.0f)<1e-5f && fabsf(co[3]-5.0f)<1e-5f, "entry1 mean = 5");

    /* U03 vision hash: same tokens -> same hash; different -> (likely) different. */
    int a[3] = {7,8,9}, b[3] = {7,8,9};
    CHECK(wubu_vision_hash(a,3) == wubu_vision_hash(b,3), "same tokens -> same hash");

    /* U04 LOOK-M: keep 2 highest-score of [0.1,0.9,0.5,0.3] -> ids 1,2. */
    float sc[4] = {0.1f,0.9f,0.5f,0.3f};
    int kp[4]; int km = wubu_lookm_keep(sc, 4, 2, kp);
    CHECK(km == 2, "keep 2");
    CHECK((kp[0]==1 && kp[1]==2), "keep ids {1,2} (top scores)");

    /* U05 agentic compaction: saliency [1,9,3,7], keep 2 -> keep ids 1,3. */
    float sal[4] = {1.0f,9.0f,3.0f,7.0f};
    char mask[4];
    int kept = wubu_agentic_compact(sal, 4, 2, mask);
    CHECK(kept == 2, "keep 2 turns");
    CHECK(mask[1]==1 && mask[3]==1 && mask[0]==0 && mask[2]==0, "keep {1,3}, compact {0,2}");

    if (failures == 0) { printf("ALL AGENTIC-KV TESTS PASSED\n"); return 0; }
    printf("%d AGENTIC-KV TEST(S) FAILED\n", failures);
    return 1;
}
