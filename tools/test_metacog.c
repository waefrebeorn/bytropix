/* test_metacog.c -- Theme JD: AGI metacognition (first batch). */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_metacog.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_metacog (JD batch 1) ===\n");

    /* JD01/JD02: profiles + self-assessment */
    {
        wubu_metacog_t m;
        CHECK(wubu_mc_init(&m, 3) == 0, "init 3 agents");
        CHECK(wubu_mc_set_competence(&m, 0, 0.9f) == 0, "set comp");
        CHECK(wubu_mc_set_competence(&m, 1, 0.5f) == 0, "set comp 2");
        CHECK(wubu_mc_set_competence(&m, 0, 1.5f) == 0 && m.competence[0] == 1.0f,
              "competence clamped");
        /* the clamp overwrote agent 0: restore the 0.9 profile */
        wubu_mc_set_competence(&m, 0, 0.9f);
        /* the assessment pulls the claimed confidence toward competence */
        float a_hi = wubu_mc_assess(&m, 0, 1.0f);   /* overclaim, comp 0.9 */
        float a_lo = wubu_mc_assess(&m, 1, 0.0f);   /* underclaim, comp 0.5 */
        CHECK(a_hi < 1.0f && a_hi > 0.9f, "overclaim pulled back");
        CHECK(a_lo > 0.0f && a_lo < 0.5f, "underclaim pulled up");
        CHECK(wubu_mc_set_competence(&m, 9, 0.5f) == -1, "OOB agent");
        CHECK(wubu_mc_init(&m, 0) == -1, "zero agents rejected");
    }

    /* JD03: ECE */
    {
        wubu_mc_ece_t e;
        wubu_mc_ece_init(&e);
        /* perfectly calibrated: conf == accuracy */
        wubu_mc_ece_feed(&e, 0.8f, 0.8f);
        wubu_mc_ece_feed(&e, 0.5f, 0.5f);
        NEAR(wubu_mc_ece_score(&e), 0.0f, 1e-5f);
        /* miscalibrated */
        wubu_mc_ece_init(&e);
        wubu_mc_ece_feed(&e, 0.9f, 0.1f);
        NEAR(wubu_mc_ece_score(&e), 0.8f, 1e-5f);
        wubu_mc_ece_init(&e);
        NEAR(wubu_mc_ece_score(&e), 0.0f, 1e-6f);
    }

    /* JD04/JD05: JOL + harness */
    {
        NEAR(wubu_mc_jol(-0.1f, -2.0f), 1.0f - expf(-1.9f), 1e-4f);
        NEAR(wubu_mc_jol(-0.5f, -0.5f), 0.0f, 1e-5f);   /* tie -> no JOL */
        CHECK(wubu_mc_harness(0.9f, 0.3f, 0.7f, 3) == 1, "high JOL -> continue");
        CHECK(wubu_mc_harness(0.5f, 0.3f, 0.7f, 3) == 2, "mid JOL -> retry");
        CHECK(wubu_mc_harness(0.5f, 0.3f, 0.7f, 0) == 0, "no retries -> stop");
        CHECK(wubu_mc_harness(0.1f, 0.3f, 0.7f, 3) == 0, "low JOL -> stop");
    }

    /* JD08/JD09 */
    {
        CHECK(wubu_mc_separated(0.9f, 0.1f, 0.3f) == 1, "monitor drift");
        CHECK(wubu_mc_separated(0.9f, 0.85f, 0.3f) == 0, "agree");
        NEAR(wubu_mc_progress(0.5f, 0.05f, 10), 1.0f, 1e-5f);  /* capped */
        NEAR(wubu_mc_progress(0.5f, 0.02f, 10), 0.7f, 1e-5f);
    }

    /* JD10/JD11 */
    {
        NEAR(wubu_mc_recalibrate(0.8f, 0.5f, 1.0f), 0.6f, 1e-5f);
        NEAR(wubu_mc_recalibrate(0.8f, 0.0f, 1.0f), 0.8f, 1e-5f);
        NEAR(wubu_mc_faithfulness(0.9f, 0.9f), 0.0f, 1e-5f);
        NEAR(wubu_mc_faithfulness(0.9f, 0.1f), 0.8f, 1e-5f);
    }

    /* JD12/JD13 */
    {
        float d[4] = { 0.9f, 0.7f, 0.5f, 0.3f };
        int passed = 0;
        CHECK(wubu_mc_eval_run(d, 4, 0.6f, 1.0f, &passed) == 4, "eval ran");
        CHECK(passed == 2, "two tasks above competence");
        wubu_metacog_t m;
        wubu_mc_init(&m, 3);
        wubu_mc_set_competence(&m, 0, 0.9f);
        wubu_mc_set_competence(&m, 1, 0.5f);
        wubu_mc_set_competence(&m, 2, 0.2f);
        CHECK(wubu_mc_delegate(&m, 0.4f) == 0, "delegate to the top agent");
    }

    /* JD14/JD16/JD17/JD18 */
    {
        wubu_metacog_t m;
        wubu_mc_init(&m, 1);
        wubu_mc_set_competence(&m, 0, 0.5f);
        wubu_mc_update_competence(&m, 0, 1.0f, 0.5f);
        NEAR(m.competence[0], 0.75f, 1e-5f);
        uint32_t tr[4] = { 0 };
        int n = 0;
        wubu_mc_trace(tr, &n, 4, 7);
        wubu_mc_trace(tr, &n, 4, 9);
        CHECK(n == 2 && tr[0] == 7 && tr[1] == 9, "trace appended");
        wubu_mc_trace(tr, &n, 4, 1);
        wubu_mc_trace(tr, &n, 4, 2);
        CHECK(n == 4, "trace capped");
        CHECK(wubu_mc_second_order(0.9f, 0.5f, 0.3f) == 1, "second-order flag");
        NEAR(wubu_mc_telemetry((float[]){ 1, 2, 3 }, 3), 2.0f, 1e-5f);
    }

    /* JD19/JD24/JD30 */
    {
        NEAR(wubu_mc_gap(0.8f, 0.5f), 0.3f, 1e-5f);
        NEAR(wubu_mc_gap(0.4f, 0.8f), -0.4f, 1e-5f);
        NEAR(wubu_mc_sample_temp(1.0f, 1.0f), 1.0f, 1e-5f);
        NEAR(wubu_mc_sample_temp(0.5f, 1.0f), 1.5f, 1e-5f);
        wubu_mc_ece_t e;
        wubu_mc_ece_init(&e);
        wubu_mc_ece_feed(&e, 0.9f, 0.1f);
        NEAR(wubu_mc_drift(&e, 0.8f), 0.0f, 1e-5f);
        NEAR(wubu_mc_drift(&e, 0.2f), 0.6f, 1e-5f);
    }

    if (failures == 0) printf("ALL METACOG TESTS PASSED\n");
    else printf("%d METACOG FAILURES\n", failures);
    return failures ? 1 : 0;
}
