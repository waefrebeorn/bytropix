/*
 * test_sys2026.c -- Q12/Q13/Q14/Q16/Q17/Q18/R02 verification.
 */
#include "wubu_sys2026.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_sys2026 (Q12/Q13/Q14/Q16/Q17/Q18/R02) ===\n");

    int ages[8] = {0,1,2,3,4,5,6,7}; /* oldest = 7 */

    /* Q12 TARDIS: 8 tokens, gpu_cap 5 -> spill 3. */
    CHECK(wubu_tardis_spill(ages, 8, 5) == 3, "over cap -> spill 3");
    CHECK(wubu_tardis_spill(ages, 3, 5) == 0, "under cap -> 0 spill");

    /* Q13 KVDrive tiers. */
    CHECK(wubu_kvdrive_tier(1, 10, 100) == 0, "recent -> GPU");
    CHECK(wubu_kvdrive_tier(50, 10, 100) == 1, "mid -> DRAM");
    CHECK(wubu_kvdrive_tier(200, 10, 100) == 2, "cold -> SSD");

    /* Q14 ScoutAttention eligibility: at cur=2, ahead=3 -> layers 2,3,4 eligible. */
    CHECK(wubu_scout_eligible(3, 2, 3) == 1, "layer in window -> eligible");
    CHECK(wubu_scout_eligible(6, 2, 3) == 0, "layer beyond window -> not");

    /* Q16 AlignedServe LCP: new [1,2,3,9], existing [1,2,9,9] + [5,5,5,5] -> max 2. */
    int newr[4] = {1,2,3,9};
    int reqs[8] = {1,2,9,9, 5,5,5,5};
    CHECK(wubu_aligned_lcp(newr, 4, reqs, 2, 4) == 2, "max shared prefix 2");

    /* Q17 CoDec share: both len 10, min 4 -> share. */
    CHECK(wubu_codec_share(10, 10, 4) == 1, "shared prefix >= min -> share");
    CHECK(wubu_codec_share(2, 10, 4) == 0, "short prefix -> no share");

    /* Q18 SparKV: high p*benefit > cost -> load. */
    CHECK(wubu_sparkv_load(0.9f, 1.0f, 0.5f) == 1, "benefit>cost -> load");
    CHECK(wubu_sparkv_load(0.1f, 1.0f, 0.5f) == 0, "low p -> skip load");

    /* R02 agentic ctx: cost fits budget -> use curated context. */
    CHECK(wubu_agentic_ctx(100.0f, 200.0f) == 1, "cost fits -> use context");
    CHECK(wubu_agentic_ctx(300.0f, 200.0f) == 0, "cost exceeds -> skip");

    if (failures == 0) { printf("ALL SYS2026 TESTS PASSED\n"); return 0; }
    printf("%d SYS2026 TEST(S) FAILED\n", failures);
    return 1;
}
