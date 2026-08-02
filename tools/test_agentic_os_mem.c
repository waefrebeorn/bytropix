/*
 * test_agentic_os_mem.c -- AD01-AD04 + AE01-AE04 verification.
 */
#include "wubu_agentic_os.h"
#include "wubu_agentic_mem.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_agentic_os_mem (AD01-AD04 + AE01-AE04) ===\n");

    /* AD01 9P capability enforcement */
    CHECK(wubu_9p_cap_allowed("/n/agent_7", "/n/agent_7/kv/layer_0") == 1, "child path allowed");
    CHECK(wubu_9p_cap_allowed("/n/agent_7", "/n/agent_8/kv") == 0, "sibling denied");
    CHECK(wubu_9p_cap_allowed("/n/agent_7", "/n/agent_7") == 1, "exact subtree root allowed");

    /* AD02 backoff */
    CHECK(wubu_backoff_ms(0, 100, 5) == 100, "attempt0 -> base");
    CHECK(wubu_backoff_ms(3, 100, 5) == 800, "attempt3 -> 800ms");
    CHECK(wubu_backoff_ms(9, 100, 5) == 3200, "capped at 2^5*100");
    CHECK(wubu_skip_if_running(1) == 1, "running -> skip");
    CHECK(wubu_skip_if_running(0) == 0, "idle -> run");

    /* AD03 checkpoint */
    wubu_checkpoint_t c; wubu_checkpoint_pack(&c, 4096L, 17);
    long s = 0; int st = 0;
    CHECK(wubu_checkpoint_resume(&c, &s, &st) == 1 && s == 4096L && st == 17, "resume restores seq/step");

    /* AD04 resource bound */
    wubu_resbound_t b = { 500, 2048, 10240 };
    CHECK(wubu_resbound_check(&b, 100, 100, 100) == 0, "within budget");
    CHECK(wubu_resbound_check(&b, 600, 100, 100) == 1, "cpu overrun flagged");
    CHECK(wubu_resbound_check(&b, 100, 3000, 100) == 2, "ram overrun flagged");
    CHECK(wubu_resbound_check(&b, 100, 100, 20000) == 4, "io overrun flagged");

    /* AE03 tier */
    CHECK(wubu_mem_tier(0.9f, 5000) == WUBU_TIER_LONGTERM, "high imp + long ttl -> longterm");
    CHECK(wubu_mem_tier(0.5f, 100)  == WUBU_TIER_SESSION, "mid -> session");
    CHECK(wubu_mem_tier(0.1f, 1)    == WUBU_TIER_WORKING, "low -> working");

    /* AE01 consolidation */
    CHECK(wubu_mem_consolidate(0.8f, 0.7f) == 1, "imp>=thresh -> consolidate");
    CHECK(wubu_mem_consolidate(0.3f, 0.7f) == 0, "imp<thresh -> keep raw");

    /* AE02 dedup */
    CHECK(wubu_mem_dedup(0.5f, 0.9f) == 2, "new higher -> keep new");
    CHECK(wubu_mem_dedup(0.9f, 0.5f) == 1, "existing higher -> keep existing");
    CHECK(wubu_mem_dedup(0.5f, 0.5f) == 0, "equal -> no dedup");

    /* AE04 retrieval score + forgetting */
    float s1 = wubu_mem_retrieval_score(1.0f, 0, 100);
    float s2 = wubu_mem_retrieval_score(1.0f, 100, 100);
    CHECK(fabsf(s1 - 1.0f) < 1e-5f, "age0 -> full importance");
    CHECK(s2 < s1, "older -> lower score (forgetting)");
    CHECK(wubu_mem_key_eq("fact:a", "fact:a") == 1, "key match");
    CHECK(wubu_mem_key_eq("fact:a", "fact:b") == 0, "key mismatch");

    if (failures == 0) { printf("ALL AGENTIC-OS/MEM TESTS PASSED\n"); return 0; }
    printf("%d AGENTIC TEST(S) FAILED\n", failures);
    return 1;
}
