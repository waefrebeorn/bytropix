/* test_metagame2.c -- Theme JD complete: the metacognition frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_metagame2.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_metagame2 (JD complete) ===\n");
    {
        float conf[3] = { 0.9f, 0.8f, 0.7f };
        int action = 0;
        CHECK(wubu_meta_regulate(conf, 3, 0.5f, &action) == 0 && action == 0, "continue");
    }
    {
        float comp[3] = { 0.3f, 0.9f, 0.5f };
        int chosen = -1;
        CHECK(wubu_meta_strategy(comp, 3, &chosen) == 0 && chosen == 1, "best strategy");
    }
    {
        float alloc = 0;
        CHECK(wubu_meta_compute(0.8f, 100.0f, &alloc) == 0, "compute alloc");
        NEAR(alloc, 80.0f, 1e-4f);
    }
    {
        char ref[64];
        CHECK(wubu_meta_reflect("hello", ref, 64) == 5, "reflection");
    }
    {
        float a[2] = { 0.9f, 0.1f }, b[2] = { 0.1f, 0.9f }, diff;
        CHECK(wubu_meta_asymmetry(a, b, 2, &diff) == 0, "asymmetry");
        NEAR(diff, sqrtf(1.28f), 1e-4f);
    }
    {
        float hist[5] = { 0.5f, 0.6f, 0.7f, 0.8f, 0.9f };
        float p = wubu_meta_progress(hist, 5, 0.1f);
        CHECK(p > 0.85f, "progress predicted upward");
    }
    CHECK(wubu_meta_audit(0.9f, 0.85f, 0.1f) == 1, "audit pass");
    CHECK(wubu_meta_audit(0.9f, 0.5f, 0.1f) == 0, "audit fail");
    {
        int found = 0;
        CHECK(wubu_meta_skill_lib("test", 5, &found) == 0 && found == 1, "skill found");
    }
    {
        int action = 0;
        CHECK(wubu_meta_reg_policy(0.6f, 0.2f, &action) == 0 && action == 2, "stop");
        CHECK(wubu_meta_reg_policy(0.4f, 0.5f, &action) == 0 && action == 1, "retry");
        CHECK(wubu_meta_reg_policy(0.1f, 0.9f, &action) == 0 && action == 0, "delegate");
    }
    NEAR(wubu_meta_energy(100, 0.5f), 50.0f, 1e-4f);
    {
        float conf[5] = { 0.8f, 0.81f, 0.79f, 0.82f, 0.78f };
        CHECK(wubu_meta_stability(conf, 5, 0.05f) == 1, "stable");
    }
    {
        float ledger[1] = { 0 };
        CHECK(wubu_meta_feedback(0.1f, ledger, 1) == 0, "feedback");
        NEAR(ledger[0], 0.1f, 1e-5f);
    }
    {
        float src[2] = { 1, 0 }, dst[2] = { 0, 1 };
        NEAR(wubu_meta_transfer(src, dst, 2), 0.0f, 1e-5f);
    }
    NEAR(wubu_meta_pass1(0.9f, 10), 0.81f, 1e-4f);
    CHECK(wubu_meta_early_stop(0.95f, 0.005f, 0.9f) == 1, "early stop");
    CHECK(wubu_meta_reg_budget(0.3f, 0.5f) == 1, "budget ok");
    {
        float comp[3] = { 0.2f, 0.3f, 0.1f };
        int delegated = 0;
        CHECK(wubu_meta_delegate(comp, 3, &delegated) == 0 && delegated == 1, "delegate");
    }
    CHECK(wubu_meta_independence(0.9f, 0.3f, 0.5f) == 1, "independent");
    CHECK(wubu_meta_independence(0.9f, 0.85f, 0.5f) == 0, "dependent");
    {
        float scores[3] = { 0.8f, 0.9f, 0.7f };
        NEAR(wubu_meta_bench(scores, 3), 0.8f, 1e-5f);
    }
    NEAR(wubu_meta_calib_cost(10.0f, 5.0f), 2.0f, 1e-5f);

    if (failures == 0) printf("ALL METAGAME2 TESTS PASSED\n");
    else printf("%d METAGAME2 FAILURES\n", failures);
    return failures ? 1 : 0;
}