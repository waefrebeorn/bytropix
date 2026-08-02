/* test_rsi.c -- Theme IV batch 1: the recursive self-improvement frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_rsi.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_rsi (IV batch 1) ===\n");

    /* IV01: the verifier gate */
    {
        int fails = 0;
        CHECK(wubu_rsi_gate(0.9f, 0.8f, &fails) == 1 && fails == 0, "pass resets");
        CHECK(wubu_rsi_gate(0.5f, 0.8f, &fails) == 0 && fails == 1, "fail counts");
    }

    /* IV02: improve the improver when the meta is the bottleneck */
    CHECK(wubu_rsi_improve_improver(0.9f, 0.4f, 0.5f) == 1, "meta bottleneck");
    CHECK(wubu_rsi_improve_improver(0.4f, 0.9f, 0.5f) == 0, "self is the issue");

    /* IV03: LADDER decomposition */
    {
        int k = 0;
        wubu_rsi_decompose(6.0f, 100.0f, 2, &k);
        CHECK(k >= 2 && k <= 8, "difficult goal splits");
        wubu_rsi_decompose(0.5f, 100.0f, 2, &k);
        CHECK(k == 1, "easy goal stays whole");
    }

    /* IV04: prompt mutation changes under low fitness */
    {
        char child[32] = { 0 };
        wubu_rsi_prompt_mutate("solve the problem", child, 32, 0.1f);
        CHECK(strlen(child) > 0, "child produced");
        CHECK(strcmp(child, "solve the problem") != 0, "mutated");
    }

    /* IV05: transfer */
    NEAR(wubu_rsi_transfer(0.8f, 1.0f), 0.8f, 1e-5f);
    CHECK(wubu_rsi_transfer(0.8f, 0.0f) < 0.8f, "dissimilar discounts");

    /* IV06: harness score */
    NEAR(wubu_rsi_harness(1.0f, 1), 1.0f, 1e-5f);
    CHECK(wubu_rsi_harness(0.5f, 0) < 1.0f, "partial harness");

    /* IV07: reflection gradient */
    {
        float w[3] = { 1, 1, 0 }, l[3] = { 0, 0, 1 }, g = 0;
        CHECK(wubu_rsi_reflect(w, l, 3, &g) == 0, "reflect");
        NEAR(g, 1.0f / 3.0f, 1e-5f);
    }

    /* IV08: soft-mellowmax is between max and mean */
    {
        float v[3] = { 1, 2, 3 };
        float mm = wubu_rsi_mellowmax(v, 3, 1.0f);
        CHECK(mm > 2.0f && mm <= 3.0f, "between mean and max");
        NEAR(wubu_rsi_mellowmax(v, 3, 100.0f), 3.0f, 0.05f);
    }

    /* IV09: the experience loop */
    {
        wubu_rsi_exp_t e = { 0, 0, 0 };
        wubu_rsi_experience(&e, 1, 1.0f);
        wubu_rsi_experience(&e, 0, 0.0f);
        CHECK(e.evals == 2 && e.wins == 1, "counts");
        CHECK(e.running > 0 && e.running < 1.0f, "running estimate");
    }

    /* IV10: synthetic gate */
    CHECK(wubu_rsi_synth(0.9f, 0.5f, 0.8f) == 1, "high quality passes");
    CHECK(wubu_rsi_synth(0.5f, 0.5f, 0.8f) == 0, "low quality rejected");

    /* IV11: weak-to-strong */
    NEAR(wubu_rsi_weak2strong(0.8f, 1.0f), 0.8f, 1e-5f);

    /* IV14: awareness calibration */
    NEAR(wubu_rsi_awareness(0.7f, 0.7f), 1.0f, 1e-5f);
    NEAR(wubu_rsi_awareness(0.7f, 0.3f), 0.6f, 1e-5f);

    /* IV15: bounded self-modification */
    NEAR(wubu_rsi_bounded_delta(2.0f, 0.5f, 1.0f), 0.5f, 1e-5f);
    NEAR(wubu_rsi_bounded_delta(2.0f, 0.5f, 0.0f), 0.0f, 1e-5f);

    /* IV16: fine-tune scheduler */
    CHECK(wubu_rsi_ft_schedule(100, 100, 0.1f) == 1, "cadence hit");
    CHECK(wubu_rsi_ft_schedule(50, 100, 0.4f) == 1, "drift trigger");
    CHECK(wubu_rsi_ft_schedule(50, 100, 0.1f) == 0, "no trigger");

    if (failures == 0) printf("ALL RSI TESTS PASSED\n");
    else printf("%d RSI FAILURES\n", failures);
    return failures ? 1 : 0;
}
