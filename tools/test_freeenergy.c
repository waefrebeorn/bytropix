/* test_freeenergy.c -- Theme IN: predictive coding / free energy. */
#include <stdio.h>
#include <math.h>
#include "wubu_freeenergy.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_freeenergy (IN01-IN07) ===\n");

    /* IN01: prediction error */
    NEAR(wubu_fe_pred_error(1.0f, 0.8f), 0.2f, 1e-6f);
    NEAR(wubu_fe_pred_error(1.0f, 1.0f), 0.0f, 1e-6f);

    /* IN04: precision weighting */
    NEAR(wubu_fe_precision_weight(0.2f, 10.0f), 2.0f, 1e-6f);
    NEAR(wubu_fe_precision_weight(0.2f, 0.0f), 0.0f, 1e-6f);
    NEAR(wubu_fe_precision_weight(0.2f, -5.0f), 0.0f, 1e-6f);

    /* IN02: free energy = -accuracy + complexity */
    NEAR(wubu_fe_free_energy(-1.0f, 0.5f), 1.5f, 1e-6f);
    NEAR(wubu_fe_free_energy(-0.1f, 0.0f), 0.1f, 1e-6f);
    NEAR(wubu_fe_free_energy(0.5f, 0.0f), 0.0f, 1e-6f);   /* clamped */
    NEAR(wubu_fe_free_energy(-1.0f, -2.0f), 1.0f, 1e-6f); /* neg complexity */

    /* IN03: expected free energy + policy prior */
    NEAR(wubu_fe_expected_free_energy(-3.0f, 0.5f), -2.5f, 1e-6f);
    {
        float G[3] = { -5.0f, -1.0f, 0.0f };
        float out[3];
        CHECK(wubu_fe_policy_prior(G, 3, 2.0f, out) == 0, "policy prior");
        /* the min-G policy dominates the softmax */
        CHECK(out[0] > 0.9f, "lowest G dominates");
        float sum = out[0] + out[1] + out[2];
        NEAR(sum, 1.0f, 1e-4f);
        CHECK(wubu_fe_policy_prior(NULL, 3, 2.0f, out) == -1, "null rejected");
    }

    /* IN05: perception step -- prediction moves toward the observation
     * scaled by the precision and the learning rate */
    NEAR(wubu_fe_percept_step(0.8f, 0.2f, 1.0f, 0.5f), 0.9f, 1e-6f);
    NEAR(wubu_fe_percept_step(0.8f, 0.2f, 10.0f, 0.5f), 1.8f, 1e-6f);
    NEAR(wubu_fe_percept_step(0.8f, 0.2f, 1.0f, 0.0f), 0.8f, 1e-6f);

    /* IN06: epistemic value = the uncertainty reduction */
    NEAR(wubu_fe_epistemic_value(1.0f, 0.3f), 0.7f, 1e-6f);
    NEAR(wubu_fe_epistemic_value(1.0f, 1.2f), 0.0f, 1e-6f); /* no gain */
    NEAR(wubu_fe_epistemic_value(0.0f, 0.0f), 0.0f, 1e-6f);

    /* IN07: free-energy-gated model selection */
    {
        const float fe[4] = { 3.0f, 1.0f, 2.0f, 0.5f };
        const float cx[4] = { 0.1f, 1.0f, 0.5f, 3.0f };
        /* complexity budget 1.0: models 0,1,2 -> min FE = 1.0 (idx 1) */
        CHECK(wubu_fe_pick_model(fe, cx, 4, 1.0f) == 1, "min FE under budget");
        /* budget 3.0 -> model 3 (FE 0.5) */
        CHECK(wubu_fe_pick_model(fe, cx, 4, 3.0f) == 3, "min FE under 3.0");
        /* budget 0.05 -> none */
        CHECK(wubu_fe_pick_model(fe, cx, 4, 0.05f) == -1, "none affordable");
        CHECK(wubu_fe_pick_model(fe, cx, 4, -1.0f) == -1, "neg budget");
    }

    if (failures == 0) printf("ALL FREE-ENERGY TESTS PASSED\n");
    else printf("%d FREE-ENERGY FAILURES\n", failures);
    return failures ? 1 : 0;
}
