/* test_pref2.c -- Theme IQ complete: the alignment frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_pref2.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_pref2 (IQ complete) ===\n");

    /* IQ02: CPO gated by the conditional score */
    {
        float easy = wubu_pref2_cpo(1.0f, -1.0f, 0.1f, 1.0f);
        float hard = wubu_pref2_cpo(1.0f, -1.0f, 0.9f, 1.0f);
        CHECK(hard > easy, "hard prompts weigh more");
    }

    /* IQ21: three-way */
    CHECK(wubu_pref2_threeway(2.0f, 0.0f, 0, 1.0f) > 0, "win/lose loss");
    NEAR(wubu_pref2_threeway(2.0f, 0.0f, 1, 1.0f), 1.0f, 1e-5f);

    /* IQ23: calibration */
    NEAR(wubu_pref2_calib(0.8f, 0.8f), 1.0f, 1e-5f);
    NEAR(wubu_pref2_calib(0.8f, 0.3f), 0.5f, 1e-5f);

    /* IQ24: conflict */
    {
        float a[3] = { 1, 2, 3 }, b[3] = { 1, 2, 3 }, c[3] = { 5, 5, 5 };
        CHECK(wubu_pref2_conflict(a, b, 3, 0.1f) == 1, "contradictory pair");
        CHECK(wubu_pref2_conflict(a, c, 3, 0.1f) == 0, "distinct pair");
    }

    /* IQ25: robustness envelope */
    NEAR(wubu_pref2_envelope(2.0f, 1.0f), 1.0f, 1e-6f);
    NEAR(wubu_pref2_envelope(0.5f, 1.0f), 0.5f, 1e-6f);

    /* IQ26: budget allocation */
    {
        float spent = 0;
        CHECK(wubu_pref2_alloc(0.9f, 1.0f, &spent) == 1, "within budget");
        CHECK(spent > 0, "spent tracked");
    }

    /* IQ27: KL anchor */
    NEAR(wubu_pref2_anchor(1.0f, 0.5f, 2.0f), 2.0f, 1e-5f);

    /* IQ29: reward trace */
    {
        float lp[4] = { -5, -4, -6, -3 }, tr[4];
        wubu_pref2_reward_trace(lp, 4, tr);
        NEAR(tr[0], 0.0f, 1e-5f);
        NEAR(tr[3], 2.0f, 1e-5f);
    }

    /* IQ30: bench */
    {
        float w[3] = { 1, 2, 3 }, l[3] = { 0, 5, 1 };
        NEAR(wubu_pref2_bench(w, l, 3), 2.0f / 3.0f, 1e-6f);
    }

    /* IQ33: augmentation */
    {
        float rej[3] = { 1, 2, 3 }, pair[3];
        wubu_pref2_augment(rej, 3, pair);
        NEAR(pair[1], 2.0f, 1e-6f);
    }

    /* IQ34: drift */
    CHECK(wubu_pref2_drift(0.8f, 0.5f, 0.2f) == 1, "drift flagged");
    CHECK(wubu_pref2_drift(0.8f, 0.7f, 0.2f) == 0, "within band");

    /* IQ35: curriculum */
    {
        float w = 0;
        wubu_pref2_curriculum(0.5f, 0.5f, &w);
        CHECK(w > 0.9f, "current band peaks");
        wubu_pref2_curriculum(0.9f, 0.5f, &w);
        CHECK(w < 0.6f, "far band dips");
    }

    /* IQ36: shaping */
    NEAR(wubu_pref2_shape(2.0f, 3.0f, 1.0f), 7.0f, 1e-5f);

    /* IQ37: batch mix */
    NEAR(wubu_pref2_batch_mix(1.0f, 0.0f, 0.5f), 0.5f, 1e-5f);

    /* IQ39: constrained decode */
    {
        float lp[3] = { 1, 2, 3 }, out[3];
        wubu_pref2_constrained(lp, 3, 2, 10.0f, out);
        NEAR(out[2], 13.0f, 1e-5f);
    }

    /* IQ41: provenance */
    {
        int t1 = 0, t2 = 0;
        wubu_pref2_provenance("user-feedback", &t1);
        wubu_pref2_provenance("user-feedback", &t2);
        CHECK(t1 == t2 && t1 != 0, "stable provenance tag");
    }

    /* IQ42: multi-turn (later turns discounted) */
    {
        float tr[3] = { 1, 1, 1 };
        CHECK(wubu_pref2_multiturn(tr, 3) < 3.0f, "recency discounted");
    }

    /* IQ43: staleness */
    NEAR(wubu_pref2_stale_weight(10.0f, 10.0f), 0.5f, 1e-4f);

    /* IQ44: quality gate */
    CHECK(wubu_pref2_quality(0.9f, 0.8f) == 1, "good pair");
    CHECK(wubu_pref2_quality(0.5f, 0.8f) == 0, "low agreement rejected");

    /* IQ45: method divergence */
    NEAR(wubu_pref2_method_div(0.5f, 0.3f), 0.2f, 1e-5f);

    /* IQ46: ensemble */
    {
        float r[3] = { 1, 2, 3 };
        NEAR(wubu_pref2_ensemble(r, 3, NULL), 2.0f, 1e-5f);
    }

    /* IQ47: health dashboard */
    NEAR(wubu_pref2_health(0.8f, 0.1f, 0.5f), 0.80f, 1e-4f);

    /* IQ49: bootstrap */
    CHECK(wubu_pref2_bootstrap(0.9f, 0.7f) == 1, "confident bootstrap");

    /* IQ51: length-robust */
    NEAR(wubu_pref2_len_robust(-10.0f, 10, 1.0f), -1.0f, 1e-5f);

    /* IQ52: confidence-scaled temp */
    NEAR(wubu_pref2_conf_temp(1.0f), 0.2f, 1e-5f);
    NEAR(wubu_pref2_conf_temp(0.0f), 1.0f, 1e-5f);

    /* IQ53: margin prediction */
    NEAR(wubu_pref2_margin_predict((float[]){ 1, 2, 3 }, 3), 2.0f, 1e-5f);

    /* IQ55: verify gate */
    CHECK(wubu_pref2_verify_gate(0.9f, 0.8f) == 1, "verified");
    CHECK(wubu_pref2_verify_gate(0.5f, 0.8f) == 0, "not promoted");

    /* IQ57: hack pre-detection */
    CHECK(wubu_pref2_hack_detect(5.0f, 1.0f, 1.0f) == 1, "hack flagged");

    /* IQ58: active selection */
    CHECK(wubu_pref2_active(0.8f, 1.0f) == 1, "uncertain + budget -> select");
    CHECK(wubu_pref2_active(0.2f, 1.0f) == 0, "certain -> skip");

    /* IQ59: entropy */
    {
        float p[3] = { 0.5f, 0.5f, 0 };
        NEAR(wubu_pref2_entropy(p, 3), logf(2.0f), 1e-4f);
    }

    /* IQ60: joint align+forget */
    NEAR(wubu_pref2_joint(1.0f, 2.0f, 0.5f), 2.0f, 1e-5f);

    /* IQ64: margin reg */
    NEAR(wubu_pref2_margin_reg(2.0f, 1.0f), 1.0f, 1e-6f);

    /* IQ66: test-time scaling */
    NEAR(wubu_pref2_tts(2.0f, 1.0f), 2.0f, 1e-5f);
    NEAR(wubu_pref2_tts(2.0f, 0.0f), 1.0f, 1e-5f);

    /* IQ67: the operator */
    {
        int promoted = 0;
        CHECK(wubu_pref2_operator(0.9f, 0.8f, &promoted) == 1 && promoted == 1,
              "health -> promote");
        wubu_pref2_operator(0.5f, 0.8f, &promoted);
        CHECK(promoted == 0, "unhealthy stays");
    }

    if (failures == 0) printf("ALL PREF2 TESTS PASSED\n");
    else printf("%d PREF2 FAILURES\n", failures);
    return failures ? 1 : 0;
}
