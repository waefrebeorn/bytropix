/*
 * test_integrate.c -- runtime policy composer verification (option c: exploit gaps).
 */
#include "wubu_integrate.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_integrate (runtime decode policy) ===\n");

    /* default policy with 512K ceiling, 32 layers. */
    wubu_decode_policy_t *p = wubu_decode_policy_default(524288, 32);
    CHECK(p != NULL, "policy created");

    wubu_decode_decision_t d;

    /* normal seqlen: oom_safe, but past stream window so eviction advised. */
    wubu_decode_policy_step(p, 1000, 0, (1<<30), 0, &d);
    CHECK(d.oom_safe == 1, "seqlen 1000 fits under 512K");
    CHECK(d.force_evict == 1, "seqlen 1000 > sink+window -> evict");
    CHECK(d.keep_budget > 0.1f, "keep_budget positive");

    /* below stream window (seqlen 100 < 516): no eviction. */
    wubu_decode_decision_t d2;
    wubu_decode_policy_step(p, 100, 0, (1<<30), 0, &d2);
    CHECK(d2.force_evict == 0, "seqlen 100 < sink+window -> no evict");

    /* near ceiling: seqlen 524287 -> next 524288 <= max -> safe. */
    wubu_decode_policy_step(p, 524287, 0, (1<<30), 0, &d);
    CHECK(d.oom_safe == 1, "seqlen 524287 -> next fits");

    /* over ceiling: seqlen 524288 -> next 524289 > max -> unsafe + evict. */
    wubu_decode_policy_step(p, 524288, 0, (1<<30), 0, &d);
    CHECK(d.oom_safe == 0, "seqlen 524288 -> next exceeds ceiling");
    CHECK(d.force_evict == 1, "eviction forced over ceiling");

    /* streaming eviction: seqlen > sink+window (4+512=516) -> force_evict. */
    wubu_decode_policy_step(p, 600, 0, (1<<30), 0, &d);
    CHECK(d.force_evict == 1, "seqlen 600 > sink+window -> evict");
    CHECK(d.elastic_evict == 600 - 516, "elastic_evict = seqlen - (sink+window)");

    /* hybrid period 4: layer 0 -> attention (recurrent=0), layer 1 -> recurrent. */
    wubu_decode_policy_set_hybrid(p, 4);
    wubu_decode_policy_step(p, 100, 0, (1<<30), 0, &d);
    CHECK(d.hybrid_recurrent == 0, "layer 0 -> attention");
    wubu_decode_policy_step(p, 100, 0, (1<<30), 1, &d);
    CHECK(d.hybrid_recurrent == 1, "layer 1 -> recurrent");

    /* PD enabled: decode_qlen below high_water -> accept. */
    wubu_decode_policy_set_pd(p, 1);
    wubu_decode_policy_step(p, 100, 3, 5, 1, &d);
    CHECK(d.pd_accept == 1, "PD decode accepts below high-water");
    wubu_decode_policy_step(p, 100, 5, 5, 1, &d);
    CHECK(d.pd_accept == 0, "PD decode full -> reject");

    wubu_decode_policy_destroy(p);

    if (failures == 0) { printf("ALL INTEGRATE TESTS PASSED\n"); return 0; }
    printf("%d INTEGRATE TEST(S) FAILED\n", failures);
    return 1;
}
