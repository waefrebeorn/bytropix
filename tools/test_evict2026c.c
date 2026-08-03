/* test_evict2026c.c -- Theme IO remainder: the final KV-eviction frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_evict2026c.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_evict2026c (IO remainder) ===\n");
    {
        float attn[5] = { 0.9f, 0.1f, 0.8f, 0.05f, 0.7f };
        int keep[5];
        CHECK(wubu_evictc_h2o(attn, 5, 0.5f, keep) == 3, "H2O retention");
        CHECK(keep[0] == 0 && keep[1] == 2 && keep[2] == 4, "heavy hitters kept");
    }
    {
        int tokens[10] = { 0 };
        CHECK(wubu_evictc_sink(tokens, 10, 2, 5) == 5, "sink+tail");
    }
    {
        float kv[4] = { 0.5f, 0.95f, -0.3f, -0.95f };
        int32_t quant[4], outlier[4];
        CHECK(wubu_evictc_kvquant(kv, 4, quant, outlier) == 4, "KVQuant");
        CHECK(outlier[1] == 950 && outlier[3] == -950, "outliers flagged");
    }
    {
        float sum[3] = { 0, 0, 0 };
        CHECK(wubu_evictc_track(sum, 1, 0.5f) == 0, "track");
        NEAR(sum[1], 0.5f, 1e-5f);
    }
    {
        float orig[3] = { 1, 2, 3 }, recon[3] = { 1.1f, 2.1f, 3.1f };
        CHECK(wubu_evictc_recon_importance(orig, recon, 3, 0.2f) == 1, "recon pass");
        CHECK(wubu_evictc_recon_importance(orig, recon, 3, 0.05f) == 0, "recon fail");
    }
    {
        float kv[5] = { 0.95f, 0.1f, -0.9f, 0.01f, 0.8f };
        int idx[5];
        CHECK(wubu_evictc_outlier(kv, 5, 0.5f, idx) == 3, "outliers");
    }
    {
        float pages[4] = { 0.9f, 0.1f, 0.8f, 0.05f };
        CHECK(wubu_evictc_page_import(pages, 4, 0.5f) == 2, "page importance");
    }
    NEAR(wubu_evictc_lsh_thresh(0.8f, 0.5f), 0.1f, 1e-5f);
    {
        int count = 0;
        wubu_evictc_proxy(100, &count);
        CHECK(count == 1, "proxy tokens");
    }
    {
        int pos[3] = { 5, 6, 7 }, new_pos[3];
        CHECK(wubu_evictc_rope_reencode(pos, 3, 2, new_pos) == 3, "re-encode");
        CHECK(new_pos[0] == 3 && new_pos[2] == 5, "positions shifted");
    }
    CHECK(wubu_evictc_audit(5.0f, 5.2f) == 1, "audit pass");
    CHECK(wubu_evictc_audit(6.0f, 5.2f) == 0, "audit fail");
    {
        int table[4] = { 0 }, evict = 0;
        CHECK(wubu_evictc_block_paged(table, 4, 16, &evict) == 0 && evict == 0, "block paged");
    }
    {
        float crit[4] = { 0.9f, 0.2f, 0.7f, 0.1f };
        int evicted[4];
        CHECK(wubu_evictc_batch(crit, 4, 0.5f, evicted) == 2, "batch eviction");
    }
    {
        float kv[4] = { 0.5f, -0.5f, 1.0f, -1.0f };
        int8_t out[4];
        CHECK(wubu_evictc_kvquant_kernel(kv, 4, out) == 4, "kvquant kernel");
    }
    {
        float sim[4] = { 0.9f, 0.1f, 0.8f, 0.2f };
        int keep[4];
        CHECK(wubu_evictc_ann(sim, 4, 0.5f, keep) == 2, "ANN retention");
    }
    {
        float draft[4] = { 0.9f, 0.1f, 0.8f, 0.2f };
        int retain[4];
        CHECK(wubu_evictc_spec(draft, 4, 0.5f, retain) == 2, "spec retain");
    }
    {
        float attn[3] = { 0.5f, 1.0f, 0.75f };
        CHECK(wubu_evictc_scaling(attn, 3, 0.5f) == 0, "scaling");
        NEAR(attn[1], 0.5f, 1e-5f);
    }
    CHECK(wubu_evictc_1m(2000000, 1000000) == 1, "1M+ context");
    {
        float a[3] = { 0.9f, 0.2f, 0.7f }, s[3] = { 0.8f, 0.1f, 0.9f };
        CHECK(wubu_evictc_hybrid(a, s, 3, 0.5f) == 2, "hybrid eviction");
    }
    {
        float v[3] = { 0.9f, 0.1f, 0.8f }, t[3] = { 0.2f, 0.1f, 0.9f };
        CHECK(wubu_evictc_mm(v, t, 3, 0.5f) == 2, "multimodal");
    }

    if (failures == 0) printf("ALL EVICT2026C TESTS PASSED\n");
    else printf("%d EVICT2026C FAILURES\n", failures);
    return failures ? 1 : 0;
}
