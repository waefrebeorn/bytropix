/*
 * test_metagame2_resource.c -- AH07/AH09/AH10/AH11 (metagame2) + AH14/AH15 (resource).
 */
#include "wubu_metagame2.h"
#include "wubu_resource.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_metagame2_resource (AH07/09/10/11/14/15) ===\n");

    /* AH07 sandbox gate */
    CHECK(wubu_sandbox_allow(0, 0, 0, 1) == 1, "no caps, tests pass -> allow");
    CHECK(wubu_sandbox_allow(1, 0, 1, 1) == 0, "needs net but denied -> block");
    CHECK(wubu_sandbox_allow(2, 1, 0, 1) == 0, "needs fs but denied -> block");
    CHECK(wubu_sandbox_allow(0, 1, 1, 0) == 0, "tests fail -> block");

    /* AH09 skill library */
    wubu_skilllib_t sk; memset(&sk, 0, sizeof sk);
    wubu_skill_add(&sk, "retry", "on fail, retry", 0.9);
    wubu_skill_add(&sk, "decompose", "split goal", 0.7);
    wubu_skill_add(&sk, "verify", "check output", 0.95);
    int top[3];
    int m = wubu_skill_topk(&sk, 2, top);
    CHECK(m == 2, "top-2 retrieved");
    CHECK(strcmp(sk.name[top[0]], "verify") == 0, "best skill = verify (0.95)");

    /* AH10 continual replay (reservoir, no forgetting of reservoir) */
    wubu_replay_t rp; long buf[4]; rp.buf = buf; rp.cap = 4; rp.n = 0; rp.seen = 0;
    int ri;
    for (long e = 0; e < 10; e++) wubu_replay_add(&rp, e, &ri);
    CHECK(rp.n <= 4, "reservoir capped at 4");
    CHECK(rp.seen == 10, "all 10 seen (no forgetting of count)");

    /* AH11 intrinsic metacognition */
    wubu_metacog_t mc; memset(&mc, 0, sizeof mc);
    /* well-calibrated: conf matches outcome */
    double c = wubu_metacog_update(&mc, 0.9, 1);  /* confident + correct -> low err */
    wubu_metacog_update(&mc, 0.1, 0);              /* unconfident + wrong -> low err */
    wubu_metacog_update(&mc, 0.8, 1);
    wubu_metacog_update(&mc, 0.2, 0);
    wubu_metacog_update(&mc, 0.95, 1);
    wubu_metacog_update(&mc, 0.05, 0);
    wubu_metacog_update(&mc, 0.7, 1);
    wubu_metacog_update(&mc, 0.3, 0);
    CHECK(mc.n >= 8, "enough samples");
    CHECK(wubu_metacog_calibrated(&mc, 0.2) == 1, "well-calibrated (err<=0.2)");
    /* miscalibrated: always 0.9 but half wrong */
    wubu_metacog_t mc2; memset(&mc2, 0, sizeof mc2);
    for (int i = 0; i < 10; i++) wubu_metacog_update(&mc2, 0.9, (i % 2));
    CHECK(wubu_metacog_calibrated(&mc2, 0.2) == 0, "miscalibrated detected (err>0.2)");

    /* AH14 resource profiler */
    CHECK(wubu_pick_tier(24.0, 7) == WUBU_TIER_FIT, "7B fits 24GB");
    CHECK(wubu_pick_tier(24.0, 70) == WUBU_TIER_NOFIT, "70B no-fit 24GB");
    CHECK(wubu_pick_tier(48.0, 70) != WUBU_TIER_NOFIT, "70B fits 48GB (Q4)");
    double tps = wubu_est_toks(1008.0, 7, 4); /* RTX4090, 7B Q4 */
    CHECK(tps > 100.0, "7B Q4 ~100+ tok/s on 4090 bandwidth");
    double tps70 = wubu_est_toks(1008.0, 70, 4);
    CHECK(tps70 < tps, "70B slower than 7B (more bytes/token)");

    /* AH15 graceful degradation */
    CHECK(wubu_degrade_tier(24.0, 70) == 34, "70B on 24GB -> degrade to 34B (largest fit)");
    CHECK(wubu_degrade_tier(48.0, 70) == 70, "70B on 48GB -> fits, no degrade");
    CHECK(wubu_degrade_tier(8.0, 70) == 14, "8GB VRAM -> 14B (Q3 fits)");
    CHECK(wubu_degrade_tier(2.0, 70) == 0, "sub-7B VRAM -> none fit");

    if (failures == 0) { printf("ALL METAGAME2-RESOURCE TESTS PASSED\n"); return 0; }
    printf("%d METAGAME2-RESOURCE TEST(S) FAILED\n", failures);
    return 1;
}
