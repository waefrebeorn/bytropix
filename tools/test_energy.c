/* test_energy.c -- Theme IJ: energy-aware inference (wubu_energy).
 * Triple-DA: edge cases + monotonicity + budget semantics. */
#include <stdio.h>
#include <math.h>
#include "wubu_energy.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_energy (IJ01-IJ07) ===\n");

    /* IJ01: energy roofline additive + zero clamps */
    NEAR(wubu_energy_estimate(1000, 0.5f, 2000, 0.01f), 520.0f, 1e-3f);
    NEAR(wubu_energy_j_per_token(2048, 1e-4f), 0.2048f, 1e-4f);
    NEAR(wubu_energy_tokens_per_joule(2048, 1e-4f), 1.0f / 0.2048f, 1e-3f);
    NEAR(wubu_energy_tokens_per_joule(0, 0), 0, 1e-6f);
    CHECK(wubu_energy_estimate(-5, 1, -3, 1) == 0, "negative energy clamped");

    /* IJ02: ledger budget enforcement */
    wubu_energy_ledger_t L;
    CHECK(wubu_energy_ledger_init(&L, 10.0f) == 0, "ledger init");
    CHECK(wubu_energy_ledger_spend(&L, 4.0f, 100) == 0, "spend under budget");
    CHECK(wubu_energy_ledger_remaining(&L) > 5.9f, "remaining ~6");
    CHECK(wubu_energy_ledger_spend(&L, 6.0f, 100) == 1, "budget exhausted flags over");
    CHECK(wubu_energy_ledger_jpt(&L) > 0.049f && wubu_energy_ledger_jpt(&L) < 0.051f,
          "avg J/token = 10/200");
    CHECK(wubu_energy_ledger_init(&L, -1) == -1, "negative budget rejected");

    /* IJ03: DVFS power-cap frequency */
    /* P(f) = P_base*(f/f_base)^3; cap at 12.5% power -> f/2 */
    NEAR(wubu_energy_freq_for_cap(10.0f, 80.0f, 100.0f), 50.0f, 1e-3f);
    NEAR(wubu_energy_freq_for_cap(200.0f, 80.0f, 100.0f), 100.0f, 1e-3f);
    CHECK(wubu_energy_freq_for_cap(0, 80, 100) == 0, "zero cap -> no freq");
    /* memory-bound (compute_frac=0): jpt flat in f */
    NEAR(wubu_energy_jpt_at_freq(1.0f, 0.0f, 100.0f, 50.0f), 1.0f, 1e-4f);
    /* compute-bound: jpt scales with the runtime (2x at half freq) */
    NEAR(wubu_energy_jpt_at_freq(1.0f, 1.0f, 100.0f, 50.0f), 2.0f, 1e-4f);
    /* mixed: 0.5 compute -> jpt = 0.5*2 + 0.5 = 1.5 */
    NEAR(wubu_energy_jpt_at_freq(1.0f, 0.5f, 100.0f, 50.0f), 1.5f, 1e-4f);
    CHECK(wubu_energy_freq_optimal(10, 80, 100, 1.0f, 0.1f) > 0, "optimal freq > 0");

    /* IJ04: energy-budget early exit */
    CHECK(wubu_energy_ledger_init(&L, 10.0f) == 0, "re-init");
    wubu_energy_ledger_spend(&L, 9.5f, 100);        /* 0.5 J left */
    CHECK(wubu_energy_should_continue(&L, 0.4f, 1.0f) == 1, "0.5 >= 0.4 keeps going");
    CHECK(wubu_energy_should_continue(&L, 0.4f, 0.5f) == 0, "needs 0.8, only 0.5 -> stop");
    wubu_energy_ledger_spend(&L, 0.2f, 10);         /* 0.3 left, over budget */
    CHECK(wubu_energy_should_continue(&L, 1.0f, 1.0f) == 0, "over budget stops");

    /* IJ05: energy-tier offload */
    /* tier B cheaper per byte -> choose B (1) */
    CHECK(wubu_energy_choose_tier(1000, 0.2f, 0.05f, 1.0f) == 1, "cheaper tier B wins");
    /* tier A cheaper -> choose A (0) */
    CHECK(wubu_energy_choose_tier(1000, 0.05f, 0.2f, 1.0f) == 0, "cheaper tier A wins");
    CHECK(wubu_energy_choose_tier(1000, 0.1f, 0.1f, 0.5f) == 0, "tie -> A");
    CHECK(wubu_energy_choose_tier(0, 0.1f, 0.2f, 1.0f) == 0, "zero bytes -> A");

    /* IJ06: spec-decoding energy break-even */
    /* accept 0.5: drafter budget = 0.5 * target */
    NEAR(wubu_energy_spec_breakeven(1.0f, 0.5f, 4), 0.5f, 1e-4f);
    /* accept 0.9: nearly everything accepted -> drafter can cost ~0.9 */
    NEAR(wubu_energy_spec_breakeven(1.0f, 0.9f, 4), 0.9f, 1e-3f);
    /* accept 0 -> the drafter must be free (0) */
    NEAR(wubu_energy_spec_breakeven(1.0f, 0.0f, 4), 0.0f, 1e-4f);
    CHECK(wubu_energy_spec_breakeven(0, 0.5f, 4) == 0, "zero target -> 0");
    /* a cheap drafter's per-accepted-token energy beats the no-spec
     * target cost (0.467 < 1.0) */
    CHECK(wubu_energy_spec_round(0.1f, 1.0f, 0.5f, 4) < 1.0f,
          "cheap drafter beats no-spec energy");
    /* an expensive drafter loses: (1.0*4 + 1)/3 = 1.67 > 1.0 */
    CHECK(wubu_energy_spec_round(1.0f, 1.0f, 0.5f, 4) > 1.0f,
          "expensive drafter loses to no-spec");

    /* IJ07: the budget-driven operator config pick */
    {
        const float jpt[4] = { 1.0f, 0.6f, 0.4f, 0.2f };
        const float tps[4] = { 50, 40, 30, 10 };
        float out = 0;
        /* min 25 tok/s + budget 0.5 -> the best affordable = 0.4 */
        int r = wubu_energy_pick_config(jpt, tps, 4, 25.0f, 0.5f, &out);
        CHECK(r == 2 && fabsf(out - 0.4f) < 1e-5f, "config 2 (0.4 jpt) wins");
        /* min 45 -> only config 0 clears the throughput gate (and it
         * is affordable under a generous budget) */
        r = wubu_energy_pick_config(jpt, tps, 4, 45.0f, 5.0f, &out);
        CHECK(r == 0, "throughput gate forces config 0");
        /* budget 0.3 -> 0.2 is the only affordable */
        r = wubu_energy_pick_config(jpt, tps, 4, 0.0f, 0.3f, &out);
        CHECK(r == 3 && fabsf(out - 0.2f) < 1e-5f, "budget gates to config 3");
        /* nothing affordable -> -1 */
        r = wubu_energy_pick_config(jpt, tps, 4, 0.0f, 0.05f, &out);
        CHECK(r == -1, "no affordable config -> -1");
    }

    if (failures == 0) printf("ALL ENERGY TESTS PASSED\n");
    else printf("%d ENERGY FAILURES\n", failures);
    return failures ? 1 : 0;
}
