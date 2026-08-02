/* test_hopfield2.c -- Theme IP: Hopfield frontier extensions. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_hopfield2.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_hopfield2 (IP frontier) ===\n");

    /* RK4/continuous-time step */
    {
        float st[2] = { 1, 2 }, fld[2] = { 0.5f, -1 }, out[2];
        CHECK(wubu_hf_rk4_step(st, fld, 2, 0.1f, out) == 0, "rk4 step");
        NEAR(out[0], 1.05f, 1e-5f);
        NEAR(out[1], 1.9f, 1e-5f);
        CHECK(wubu_hf_rk4_step(NULL, fld, 2, 0.1f, out) == -1, "null rejected");
    }

    /* manifold reorganization */
    {
        float p[2] = { 1, 0 }, c[2] = { 0, 1 }, out[2];
        wubu_hf_manifold_shift(p, c, 2, 1.0f, out);
        /* (1,1) normalized */
        NEAR(out[0], 1.0f / sqrtf(2), 1e-5f);
        NEAR(out[1], 1.0f / sqrtf(2), 1e-5f);
    }

    /* federated many-to-one binding */
    {
        float cues[4] = { 1, 0,  0, 1 };   /* two cues, 2-d */
        float outs[4] = { 10, 20,  30, 40 };
        float out[2];
        int best = -1;
        CHECK(wubu_hf_federated_bind(cues, outs, 2, 2, (float[]){ 0.9f, 0.1f }, &best, out) == 1,
              "federated bind found");
        CHECK(best == 0 && out[0] == 10.0f && out[1] == 20.0f, "closest cue bound");
    }

    /* spectral capacity + separation */
    {
        float c_low = wubu_hf_spectral_capacity(8, 0.5f, 0.9f);
        float c_hi = wubu_hf_spectral_capacity(8, 0.5f, 0.1f);
        CHECK(c_hi > c_low, "low spectral saturation -> higher capacity");
        float X[8] = { 1, 0, 0, 0,  0, 1, 0, 0 };
        NEAR(wubu_hf_separation(X, 2, 4), sqrtf(2.0f), 1e-5f);
    }

    /* write scheduling + rehearsal */
    {
        CHECK(wubu_hf_should_store(0.9f, 0.5f, 10) == 1, "novel -> store");
        CHECK(wubu_hf_should_store(0.1f, 0.5f, 10) == 0, "redundant -> skip");
        CHECK(wubu_hf_should_store(0.9f, 0.5f, 0) == 0, "full -> skip");
        NEAR(wubu_hf_rehearse(1.0f, 0.5f, 2.0f), 2.0f, 1e-6f);
        NEAR(wubu_hf_rehearse(1.0f, -1.0f, 2.0f), 1.0f, 1e-6f);
    }

    /* beta annealing + denoise quality + decay schedule */
    {
        NEAR(wubu_hf_beta_anneal(8.0f, 1.0f, 0, 10), 8.0f, 1e-5f);
        NEAR(wubu_hf_beta_anneal(8.0f, 1.0f, 5, 10), 4.5f, 1e-5f);
        NEAR(wubu_hf_beta_anneal(8.0f, 1.0f, 10, 10), 1.0f, 1e-5f);
        NEAR(wubu_hf_denoise_quality(1.0f, 4.0f), 1.0f - 0.5f * expf(-4.0f), 1e-4f);
        NEAR(wubu_hf_denoise_quality(0.0f, 4.0f), 0.0f, 1e-6f);
        NEAR(wubu_hf_decay_schedule(10.0f, 1.0f), 20.0f, 1e-5f);
        NEAR(wubu_hf_decay_schedule(10.0f, 0.0f), 10.0f, 1e-5f);
    }

    /* context gate + partial overlap */
    {
        NEAR(wubu_hf_context_gate((float[]){ 1, 0 }, (float[]){ 1, 0 }, 2), 1.0f, 1e-5f);
        NEAR(wubu_hf_context_gate((float[]){ 1, 0 }, (float[]){ 0, 1 }, 2), 0.0f, 1e-5f);
        uint8_t known[2] = { 1, 0 };   /* only dim 0 known */
        NEAR(wubu_hf_partial_overlap((float[]){ 1, 0 }, (float[]){ 1, 1 }, 2, known), 1.0f, 1e-5f);
        NEAR(wubu_hf_partial_overlap((float[]){ 0, 1 }, (float[]){ 1, 1 }, 2, known), 0.0f, 1e-5f);
    }

    /* interference + orthogonalization */
    {
        float a[2] = { 1, 0 }, b[2] = { 1, 1 };
        NEAR(wubu_hf_interference(a, b, 2), 1.0f / sqrtf(2), 1e-5f);
        CHECK(wubu_hf_orthogonalize(a, 2, b) == 1, "orthogonalized");
        NEAR(wubu_hf_interference(a, b, 2), 0.0f, 1e-5f);
    }

    /* episodic weight + tool select */
    {
        NEAR(wubu_hf_episodic_weight(1.0f, 0, 10.0f), 1.0f, 1e-6f);
        NEAR(wubu_hf_episodic_weight(1.0f, 10, 10.0f), 0.5f, 1e-4f);
        float tools[6] = { 1, 0, 0,  0, 1, 0 };
        CHECK(wubu_hf_tool_select(tools, 2, 3, (float[]){ 0, 1, 0 }) == 1, "tool 1");
        CHECK(wubu_hf_tool_select(tools, 2, 3, (float[]){ 1, 0, 0 }) == 0, "tool 0");
    }

    if (failures == 0) printf("ALL HOPFIELD2 TESTS PASSED\n");
    else printf("%d HOPFIELD2 FAILURES\n", failures);
    return failures ? 1 : 0;
}
