/* test_fuzz2.c -- Theme IX complete: the robustness frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_fuzz2.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_fuzz2 (IX complete) ===\n");
    NEAR(wubu_fz2_tradeoff(0.9f, 0.5f, 0.5f), 0.7f, 1e-5f);
    CHECK(wubu_fz2_heal(5, 5) == 1, "stall recovery");
    CHECK(wubu_fz2_heal(2, 5) == 0, "healthy fuzzer");
    NEAR(wubu_fz2_signal(1, 10), 0.9f, 1e-5f);
    CHECK(wubu_fz2_schema("deep", 3, 5) == 1, "schema depth ok");
    CHECK(wubu_fz2_schema("deep", 9, 5) == 0, "too deep rejected");
    CHECK(wubu_fz2_depth((int[]){ 1, 1, 0 }, 3, 2) == 1, "layers engaged");
    CHECK(wubu_fz2_delta(0.6f, 0.8f, 0.1f) == 1, "regression flagged");
    CHECK(wubu_fz2_delta(0.75f, 0.8f, 0.1f) == 0, "within band");
    NEAR(wubu_fz2_coverage(80, 100), 0.8f, 1e-6f);
    NEAR(wubu_fz2_fp(2, 100), 0.02f, 1e-5f);
    CHECK(wubu_fz2_leak("the api key abc123", "abc123") == 1, "leak found");
    CHECK(wubu_fz2_leak("safe text", "abc123") == 0, "no leak");
    CHECK(wubu_fz2_energy(100, 0.5f, 60.0f) == 1, "within budget");
    CHECK(wubu_fz2_energy(100, 0.5f, 40.0f) == 0, "over budget");
    {
        char out[32];
        wubu_fz2_canon("  HELLO\t\tWORLD\r\n", out, 32);
        CHECK(strcmp(out, "hello world\n") == 0, "canonicalized");
    }
    CHECK(wubu_fz2_diff("abc", "abc", 1.0f) == 1, "identical");
    CHECK(wubu_fz2_diff("abc", "xyz", 0.9f) == 0, "different");
    CHECK(wubu_fz2_repair(0.9f, 0.5f) == 1, "weak guardrail repaired");
    NEAR(wubu_fz2_harness(1, 10), 0.9f, 1e-5f);
    CHECK(wubu_fz2_anomaly((uint32_t[]){ 100, 2, 3 }, 3, 3, 2) == 1,
          "anomaly detected");
    CHECK(wubu_fz2_redundant((int[]){ 1, 1, 1 }, 3, 2) == 1, "redundant");
    CHECK(wubu_fz2_degraded(1, 0) == 1, "degraded-but-safe");
    CHECK(wubu_fz2_ci(0.1f, 0.2f) == 1, "CI gate passes");
    {
        char out[32];
        int n = wubu_fz2_gen("attack", out, 32, 7);
        CHECK(n == 8 && out[7] == '7', "generated variant");
    }
    CHECK(wubu_fz2_attrib((float[]){ 0.9f, 0.2f, 0.8f }, 3) == 1, "weakest layer");
    CHECK(wubu_fz2_workers(10, 4) == 4, "parallel workers");
    CHECK(wubu_fz2_sla(0.9f, 0.8f) == 1, "SLA met");
    CHECK(wubu_fz2_verifier(1, 1) == 1, "fuzz feeds verifier");
    {
        int count = 0;
        wubu_fz2_debt((float[]){ 0.9f, 0.2f, 0.7f }, 3, 0.5f, &count);
        CHECK(count == 2, "two debts tracked");
    }
    CHECK(wubu_fz2_entropy_guard((uint32_t[]){ 50, 50, 0 }, 3, 0.5f) == 1,
          "entropy spike flagged");
    NEAR(wubu_fz2_transfer(0.8f, 0.6f), 0.75f, 1e-4f);
    NEAR(wubu_fz2_def_sampling(1.0f, 1.0f), 0.7f, 1e-5f);

    if (failures == 0) printf("ALL FUZZ2 TESTS PASSED\n");
    else printf("%d FUZZ2 FAILURES\n", failures);
    return failures ? 1 : 0;
}
