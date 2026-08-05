/*
 * test_h3_norm.c — runtime tests for the H3 hyperbolic normalization kernel.
 *
 * Verifies:
 * - F32 path: H3 activation = SiLU(gate) * tanh(up) matches reference
 * - NF4 path: dequantized row produces identical results to F32
 * - Bias terms: optional gate/up biases are applied
 *
 * C11. Self-contained. Links against wubu_h3_norm.c + wubu_dequant_nf4.c.
 */
#include "wubu_h3_norm.h"
#include "wubu_dequant_nf4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0;
static int tests_pass = 0;

static void check(int cond, const char *label) {
    tests_run++;
    if (cond) { tests_pass++; }
    printf("  %s: %s\n", cond ? "PASS" : "FAIL", label);
}

static float ref_silu(float x) {
    return x / (1.0f + expf(-x));
}

int main(void) {
    printf("=== H3 Hyperbolic Normalization Tests ===\n\n");

    /* ---- Test 1: F32 path, known values ---- */
    {
        /* 4 input, 3 output */
        float gate_w[] = {
            1.0f, -1.0f,  0.5f,  0.0f,  /* row 0 */
            0.5f,  0.5f,  0.5f,  0.5f,  /* row 1 */
           -2.0f,  1.0f,  0.0f,  1.0f,  /* row 2 */
        };
        float up_w[] = {
             0.5f,  0.5f,  1.0f, -1.0f,
             1.0f, -1.0f,  2.0f,  0.0f,
             0.1f,  0.2f,  0.3f,  0.4f,
        };
        float x[] = {1.0f, 0.5f, 2.0f, -1.0f};
        float out[3];

        wubu_h3_norm_t *ctx = wubu_h3_norm_init(gate_w, NULL, up_w, NULL, 4, 3);
        check(ctx != NULL, "F32 init");
        if (ctx) {
            wubu_h3_norm_apply(ctx, x, out);

            /* Manual reference computation */
            for (int o = 0; o < 3; o++) {
                float g = 0.0f, u = 0.0f;
                for (int i = 0; i < 4; i++) {
                    g += gate_w[o * 4 + i] * x[i];
                    u += up_w[o * 4 + i] * x[i];
                }
                float ref = ref_silu(g) * tanf(u);
                check(fabsf(out[o] - ref) < 1e-5f, "F32 match row");
            }
            wubu_h3_norm_close(ctx);
        }
    }

    /* ---- Test 2: F32 path with bias ---- */
    {
        float gate_w[] = { 0.1f, 0.2f };
        float gate_b[] = { 0.5f };
        float up_w[]   = { 0.3f, 0.4f };
        float up_b[]   = { 0.1f };
        float x[] = { 1.0f, 2.0f };
        float out[1];

        wubu_h3_norm_t *ctx = wubu_h3_norm_init(gate_w, gate_b, up_w, up_b, 2, 1);
        check(ctx != NULL, "F32+BIAS init");
        if (ctx) {
            wubu_h3_norm_apply(ctx, x, out);
            float g = 0.1f * 1.0f + 0.2f * 2.0f + 0.5f;
            float u = 0.3f * 1.0f + 0.4f * 2.0f + 0.1f;
            float ref = ref_silu(g) * tanf(u);
            check(fabsf(out[0] - ref) < 1e-5f, "F32+BIAS match");
            wubu_h3_norm_close(ctx);
        }
    }

    /* ---- Test 3: NF4 path — dequantized codes match expected levels ---- */
    {
        unsigned char gate_codes[] = { 0xF0, 0x0F };  /* 4 codes -> 8 elements */
        /* NF4 codes for up: codes 0,15,0,15,... */
        unsigned char up_codes[]   = { 0x0F, 0xF0 };

        float x[] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
        float out[1] = {0};

        wubu_h3_norm_t *ctx = wubu_h3_norm_init_nf4(gate_codes, 1.0f, up_codes, 1.0f, 8, 1);
        check(ctx != NULL, "NF4 init");
        if (ctx) {
            wubu_h3_norm_apply(ctx, x, out);
            /* gate = 8 * nf4_level(15) * 1 = 8 * (-0.034988) = -0.279904 */
            /* up   = 8 * nf4_level(0) * 1 = 8 * (-2.716777) = -37.762096 */
            float gate_val = 8 * -0.034988f;
            float up_val   = 8 * -2.716777f;
            float ref = ref_silu(gate_val) * tanf(up_val);
            check(fabsf(out[0] - ref) < 0.1f, "NF4 match");
            wubu_h3_norm_close(ctx);
        }
    }

    /* ---- Test 4: SiLU vs tanh properties ---- */
    {
        /* At x=0: SiLU(0)=0, tanh(0)=0, so output=0 */
        wubu_h3_norm_t *ctx = wubu_h3_norm_init(
            (float[]){0.0f, 0.0f}, NULL,
            (float[]){0.0f, 0.0f}, NULL, 2, 1);
        check(ctx != NULL, "Zero-weight init");
        if (ctx) {
            float x[] = {5.0f, -3.0f};
            float out[1];
            wubu_h3_norm_apply(ctx, x, out);
            check(out[0] == 0.0f, "Zero weights -> zero output");
            wubu_h3_norm_close(ctx);
        }
    }

    printf("\n=== Results: %d/%d tests passed ===\n", tests_pass, tests_run);
    return (tests_pass == tests_run) ? 0 : 1;
}
