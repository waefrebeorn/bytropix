/* test_evict2026b.c -- Theme IO batch 2: the frontier infrastructure. */
#include <stdio.h>
#include <string.h>
#include "wubu_evict2026b.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)
#include <math.h>

int main(void)
{
    printf("=== test_evict2026b (IO batch 2) ===\n");

    /* IO31: normalization is scale-free + clamped */
    NEAR(wubu_ev_norm(5.0f, 0.0f, 10.0f), 0.5f, 1e-6f);
    NEAR(wubu_ev_norm(-3.0f, 0.0f, 10.0f), 0.0f, 1e-6f);
    NEAR(wubu_ev_norm(99.0f, 0.0f, 10.0f), 1.0f, 1e-6f);

    /* IO37: sink reserve protects the first-k */
    CHECK(wubu_ev_sink_reserve(0, 4) == 1 && wubu_ev_sink_reserve(3, 4) == 1,
          "sink positions protected");
    CHECK(wubu_ev_sink_reserve(4, 4) == 0, "past the sink is evictable");

    /* IO40: batch grouping makes one-shot discard sets */
    {
        int drop[8] = { 1, 1, 0, 1, 1, 1, 0, 1 };
        int bs[4], bc[4];
        int nb = wubu_ev_batch_groups(drop, 8, 2, bs, bc, 4);
        CHECK(nb == 4, "four discard runs: [0,1] [3,4] [5] [7]");
        CHECK(bc[0] == 2 && bc[1] == 2 && bc[2] == 1 && bc[3] == 1,
              "run lengths respected");
    }

    /* IO41: max-pooling clusters the context */
    {
        float attn[8] = { 0.1f, 0.9f, 0.2f, 0.8f, 0.3f, 0.7f, 0.4f, 0.6f };
        float out[4];
        int n = wubu_ev_pool(attn, 8, 2, out, 4);
        CHECK(n == 4, "pooled into 4");
        NEAR(out[0], 0.9f, 1e-6f);
        NEAR(out[1], 0.8f, 1e-6f);
    }

    /* IO42: the priority queue evicts the min-score first */
    {
        wubu_ev_pq_t q;
        CHECK(wubu_ev_pq_init(&q, 8) == 0, "pq init");
        wubu_ev_pq_push(&q, 0.5f, 3);
        wubu_ev_pq_push(&q, 0.1f, 7);
        wubu_ev_pq_push(&q, 0.9f, 1);
        float s; int i;
        CHECK(wubu_ev_pq_pop_min(&q, &s, &i) == 0 && i == 7, "min popped first");
        CHECK(wubu_ev_pq_pop_min(&q, &s, &i) == 0 && i == 3, "second min");
        wubu_ev_pq_free(&q);
        CHECK(wubu_ev_pq_pop_min(&q, &s, &i) == -1, "freed pq refuses");
    }

    /* IO48: dual-score fusion blends importance + novelty */
    NEAR(wubu_ev_dual(1.0f, 0.0f, 0.5f), 0.5f, 1e-6f);
    NEAR(wubu_ev_dual(0.0f, 1.0f, 0.25f), 0.75f, 1e-6f);

    /* IO45: the decision cache reuses scores */
    {
        wubu_ev_cache_t c = { 0, 0, 0 };
        NEAR(wubu_ev_cache_get(&c, 5, 0.42f), 0.42f, 1e-6f);
        wubu_ev_cache_put(&c, 5, 0.88f);
        NEAR(wubu_ev_cache_get(&c, 5, 0.42f), 0.88f, 1e-6f);
        NEAR(wubu_ev_cache_get(&c, 6, 0.42f), 0.42f, 1e-6f);
    }

    /* IO33: the tier assignment */
    CHECK(wubu_ev_tier(0.9f, 0.7f, 0.4f) == 0, "hot -> RAM");
    CHECK(wubu_ev_tier(0.5f, 0.7f, 0.4f) == 1, "warm -> DRAM");
    CHECK(wubu_ev_tier(0.1f, 0.7f, 0.4f) == 2, "cold -> NVMe");

    /* IO61: compaction defragments the retained pages */
    {
        int retain[6] = { 1, 0, 1, 1, 0, 1 };
        int out[4];
        int n = wubu_ev_compact(retain, 6, out, 4);
        CHECK(n == 4 && out[0] == 0 && out[1] == 2 && out[2] == 3 && out[3] == 5,
              "survivors moved down in order");
    }

    /* IO60: the policy selector by profile */
    CHECK(wubu_ev_policy_select(0.9f, 0.1f) == 1, "head-skew -> Ada-KV");
    CHECK(wubu_ev_policy_select(0.1f, 0.9f) == 2, "block-skew -> LSH");
    CHECK(wubu_ev_policy_select(0.3f, 0.3f) == 0, "generic importance");

    /* IO63: the per-layer budget governor */
    CHECK(wubu_ev_layer_budget(0, 4, 40) == 10, "uniform share");
    CHECK(wubu_ev_layer_budget(9, 4, 40) == 0, "OOB layer -> 0");

    /* IO34/IO67: the ledger telemetry */
    {
        wubu_ev_ledger_t l = { 0, 0, 0 };
        wubu_ev_ledger_record(&l, 3, 7);
        wubu_ev_ledger_record(&l, 1, 1);
        CHECK(l.dropped == 4 && l.retained == 8, "ledger counts");

        CHECK(wubu_ev_pool(NULL, 8, 2, NULL, 4) == -1, "null pool");
        CHECK(wubu_ev_pq_init(NULL, 4) == -1, "null pq init");
    }

    if (failures == 0) printf("ALL EVICT2026B TESTS PASSED\n");
    else printf("%d EVICT2026B FAILURES\n", failures);
    return failures ? 1 : 0;
}
