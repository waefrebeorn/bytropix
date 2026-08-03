/* test_hybrid.c -- Theme JA complete: the hybrid-attention frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_hybrid.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_hybrid (JA complete) ===\n");
    {
        float a[2] = { 1, 2 }, s[2] = { 3, 4 }, out[2];
        wubu_hyb_falcon(a, s, 2, 0.5f, out);
        NEAR(out[0], 2.0f, 1e-5f);
        NEAR(out[1], 3.0f, 1e-5f);
    }
    {
        float x[4] = { 1, 2, 3, 4 };
        float attn[4], ssm[4];
        CHECK(wubu_hyb_hymba(x, 4, 2, attn, ssm) == 0, "Hymba");
    }
    {
        float x[2] = { 1, -1 }, out[2];
        wubu_hyb_qwen(x, 2, 2.0f, out);
        CHECK(out[0] > 0, "GDN positive");
    }
    NEAR(wubu_hyb_ssm_energy(57000, 1.0f), logf(2.0f), 1e-5f);
    {
        float score = 0;
        CHECK(wubu_hyb_pareto(0.9f, 0.1f, &score) == 0, "pareto");
        CHECK(score > 1, "good pareto");
    }
    {
        float comp = 0;
        wubu_hyb_recall_comp(0.6f, 0.9f, &comp);
        NEAR(comp, 0.75f, 1e-5f);
    }
    CHECK(wubu_hyb_layer_pos(0, 10) == 0, "attention layer");
    CHECK(wubu_hyb_layer_pos(7, 10) == 1, "SSM layer");
    CHECK(wubu_hyb_receptive(4, 512) == 516, "receptive field");
    CHECK(wubu_hyb_kv_budget(100, 0, 100) == 100, "KV budget");
    CHECK(wubu_hyb_decode_sched(3, 10, 0.5f) == 0, "attn decode");
    CHECK(wubu_hyb_decode_sched(7, 10, 0.5f) == 1, "SSM decode");
    NEAR(wubu_hyb_prefill_speed(57000, 1.0f, 0.1f), 0.1f * logf(58.0f), 1e-4f);
    CHECK(wubu_hyb_parity(0.94f, 0.95f) == 1, "parity met");
    CHECK(wubu_hyb_parity(0.8f, 0.95f) == 0, "parity missed");
    NEAR(wubu_hyb_energy_model(1000, 0.5f), 500.0f, 1e-4f);
    CHECK(wubu_hyb_stream(100, 200, 500) == 1, "streaming ok");
    CHECK(wubu_hyb_stream(400, 200, 500) == 0, "streaming over");
    CHECK(wubu_hyb_stability(0.5f, 0.5f, 1.0f) == 1, "stable");
    CHECK(wubu_hyb_stability(1.5f, 0.5f, 1.0f) == 0, "attn unstable");
    NEAR(wubu_hyb_reasoning(0.9f, 100000), 0.9f * (1.0f - 1.0f), 1e-5f);
    CHECK(wubu_hyb_cotrain(0.001f, 0.0005f, 0.5f) == 1, "co-train ok");
    CHECK(wubu_hyb_quant((float[]){ 0.5f, -0.5f }, 2, 4) == 2, "quantized");
    {
        long total = 0;
        CHECK(wubu_hyb_unified_cache(100, 200, &total) == 0 && total == 300, "unified");
    }
    {
        float d[3] = { 1, 2, 3 }, v[3] = { 1, 2, 0 };
        CHECK(wubu_hyb_spec_decode(d, v, 3) == 2, "spec decode");
    }

    if (failures == 0) printf("ALL HYBRID TESTS PASSED\n");
    else printf("%d HYBRID FAILURES\n", failures);
    return failures ? 1 : 0;
}
