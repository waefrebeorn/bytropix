/*
 * test_kv_budget.c -- L18/L19/N03/N17 verification.
 */
#include "wubu_kv_budget.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_kv_budget (L18/L19/N03/N17) ===\n");

    /* L18 layer-wise: deeper layers get more, mean ~ base. */
    float b0 = wubu_layer_kv_budget(0, 32, 1.0f, 0.5f, 1.5f);
    float b31 = wubu_layer_kv_budget(31, 32, 1.0f, 0.5f, 1.5f);
    CHECK(b0 < b31, "deeper layer gets larger budget");
    float sum = 0; for (int i = 0; i < 32; i++) sum += wubu_layer_kv_budget(i, 32, 1.0f, 0.5f, 1.5f);
    /* mean is normalized to ~base unless a clamp/rescale kicks in; allow 10% */
    CHECK(fabs(sum / 32.0f - 1.0f) < 0.10f, "mean budget ~= base(1.0)");
    CHECK(b0 > 0.0f && b0 <= 2.0f && b31 > 0.0f && b31 <= 2.0f, "budgets in (0,2]");
    CHECK(wubu_layer_kv_budget(-1, 32, 1.0f, 0.5f, 1.5f) > 0.0f, "OOB layer -> base");
    CHECK(wubu_layer_kv_budget(0, 0, 1.0f, 0.5f, 1.5f) > 0.0f, "L<=0 -> base");
    /* L19 adaptive sink: peaky (e=0) -> max, uniform (e=1) -> min. */
    int sp = wubu_adaptive_sink(0.0f, 2, 8);
    int su = wubu_adaptive_sink(1.0f, 2, 8);
    CHECK(sp == 8, "peaky attention -> max sinks (8)");
    CHECK(su == 2, "uniform attention -> min sinks (2)");
    CHECK(wubu_adaptive_sink(0.5f, 2, 8) == 5, "mid entropy -> 5");

    /* N03 scheme bits: KV-bound (b*<1) -> lo; weight-bound -> hi. */
    CHECK(wubu_kv_scheme_bits(0.3, 2, 16) == 2, "KV-bound -> lo bits");
    CHECK(wubu_kv_scheme_bits(50.0, 2, 16) == 16, "weight-bound -> hi bits");
    CHECK(wubu_kv_scheme_bits(1.0, 2, 16) == 16, "b*=1 boundary -> hi");

    /* N17 forecast: matches capacity_wall formula. */
    double f = wubu_kv_forecast(32, 8, 128, 16, 1, 4096);
    CHECK(fabs(f - 536870912.0) < 1.0, "forecast @4k = 512MB");
    CHECK(wubu_kv_forecast(32, 8, 128, 16, 0, 100) == 0.0, "batch 0 -> 0");

    if (failures == 0) { printf("ALL KV-BUDGET TESTS PASSED\n"); return 0; }
    printf("%d KV-BUDGET TEST(S) FAILED\n", failures);
    return 1;
}
