/* test_ternary.c -- Theme JC complete: the ternary/1.58-bit frontier. */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "wubu_ternary.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)
#define NEAR(a, b, t) CHECK(fabsf((a) - (b)) < (t), #a " ~= " #b)

int main(void)
{
    printf("=== test_ternary (JC complete) ===\n");
    {
        float w[4] = { 0.5f, -0.5f, 1.0f, -1.0f };
        int8_t out[4];
        CHECK(wubu_ternary_qat(w, 4, 1.0f, out) == 4, "QAT");
        CHECK(out[0] > 0 && out[1] < 0 && out[2] > 0 && out[3] < 0, "ternary signs");
    }
    {
        float alpha = 0;
        CHECK(wubu_ternary_schedule(5, 10, 100, &alpha) == 0 && alpha == 1.0f, "warm-up");
        CHECK(wubu_ternary_schedule(50, 10, 100, &alpha) == 0 && alpha < 1.0f, "quantizing");
    }
    NEAR(wubu_ternary_reg(1.0f, 0.5f), 0.25f, 1e-5f);
    {
        int8_t w[2] = { 1, -1 };
        float x[2] = { 0.5f, 0.5f }, out;
        CHECK(wubu_ternary_infer(w, 2, x, &out) == 0, "infer");
        NEAR(out, 0.0f, 1e-5f);
    }
    CHECK(wubu_ternary_twophase(5, 10, 20) == 0, "phase 0");
    CHECK(wubu_ternary_twophase(15, 10, 20) == 1, "phase 1");
    CHECK(wubu_ternary_twophase(35, 10, 20) == 2, "phase 2");
    {
        float bits = 0;
        CHECK(wubu_ternary_layer_prec(0, 10, &bits) == 0 && bits == 8.0f, "early layer");
        CHECK(wubu_ternary_layer_prec(9, 10, &bits) == 0 && bits == 4.0f, "late layer");
    }
    {
        float act[4] = { 1, 2, 3, 4 };
        float scale = 0;
        CHECK(wubu_ternary_act_aware(act, 4, &scale) == 0, "act-aware");
        NEAR(scale, 32.25f, 1.0f);
    }
    {
        float bits = 0;
        CHECK(wubu_ternary_curriculum(0, 100, &bits) == 0 && bits == 8.0f, "curriculum start");
        CHECK(wubu_ternary_curriculum(100, 100, &bits) == 0 && bits == 2.0f, "curriculum end");
    }
    {
        int8_t w[4] = { 1, -1, 1, -1 };
        float x[4] = { 0.5f, 0.5f, 0.5f, 0.5f }, out;
        CHECK(wubu_ternary_gemv(w, 4, x, &out) == 0, "GEMV");
    }
    {
        float w[4] = { 0.5f, -0.5f, 1.0f, -1.0f };
        int8_t out[4];
        CHECK(wubu_ternary_qat_2bit(w, 4, out) == 4, "2-bit QAT");
    }
    {
        float grad[2] = { 1, 2 };
        int8_t q[2] = { 1, -1 };
        float out_grad[2];
        CHECK(wubu_ternary_grad(grad, q, 2, out_grad) == 2, "ST estimator");
        NEAR(out_grad[0], 1.0f, 1e-5f);
    }
    {
        float kv[4] = { 1, -1, 0.5f, -0.5f };
        int32_t out[4];
        CHECK(wubu_ternary_kv_qat(kv, 4, 4, out) == 4, "KV QAT");
    }
    CHECK(wubu_ternary_transition(1.58f, 1.58f, 0.01f) == 1, "transition reached");
    NEAR(wubu_ternary_energy(100, 0.5f), 50.0f, 1e-4f);
    CHECK(wubu_ternary_finetune((float[]){1,2}, 2, 4, 10) == 1, "finetune ok");
    NEAR(wubu_ternary_ablation(4, 0.9f), 0.45f, 1e-5f);
    CHECK(wubu_ternary_robust((float[]){1,2}, 2, 0.05f) == 1, "robust");
    {
        float w[2] = { 0.5f, -0.5f }, aligned[2];
        CHECK(wubu_ternary_align(w, 2, aligned) == 2, "aligned");
    }
    {
        float w[2] = { 0.5f, -0.5f };
        int bits[2] = { 4, 8 };
        int8_t out[2];
        CHECK(wubu_ternary_mixed(w, 2, bits, out) == 2, "mixed");
    }
    NEAR(wubu_ternary_eval((float[]){1,2}, (float[]){3,4}, 2, 2), 11.0f, 1e-5f);

    if (failures == 0) printf("ALL TERNARY TESTS PASSED\n");
    else printf("%d TERNARY FAILURES\n", failures);
    return failures ? 1 : 0;
}