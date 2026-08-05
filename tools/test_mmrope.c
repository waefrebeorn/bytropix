/*
 * test_mmrope.c — tests for 3D MM-RoPE kernel.
 *
 * Verifies:
 * - init succeeds with divisible head_dim (12 -> 3 segments of 4)
 * - reject head_dim not divisible by 3 (e.g. 10)
 * - reject head_dim not even (e.g. 9)
 * - zero position -> no rotation applied (identity)
 * - basic rotation applied for non-zero position
 * - NULL safety
 */
#include "wubu_mmrope.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0;
static int tests_pass = 0;
static int tests_fail = 0;

static void check(int cond, const char *name) {
    tests_run++;
    if (cond) { tests_pass++; printf("  PASS: %s\n", name); }
    else      { tests_fail++; printf("  FAIL: %s\n", name); }
}

/* Standard 2D RoPE reference: rotate [x0, x1] by freq */
static void ref_rotate(float *x0, float *x1, float pos, float theta, int d, int i) {
    float freq = pos / powf(theta, (float)(2 * i) / (float)d);
    float c = cosf(freq);
    float s = sinf(freq);
    float a = *x0, b = *x1;
    *x0 = a * c - b * s;
    *x1 = a * s + b * c;
}

int main(void) {
    printf("=== 3D MM-RoPE Tests ===\n\n");

    /* --- Test 1: Init + basic properties --- */
    {
        int pos_t[4] = {0, 1, 2, 3};
        int pos_h[4] = {0, 0, 1, 1};
        int pos_w[4] = {0, 1, 0, 1};
        wubu_mmrope_t *ctx = wubu_mmrope_init(12, 10000.0f, 10000.0f, 10000.0f,
                                              4, 2, 2, pos_t, pos_h, pos_w);
        check(ctx != NULL, "Init head_dim=12 (div by 3, even)");

        /* Apply to a test tensor */
        float qk[4 * 1 * 12];  /* seq_len=4, n_heads=1, head_dim=12 */
        for (int i = 0; i < 48; i++) qk[i] = (float)(i + 1);

        wubu_mmrope_apply(ctx, qk, 4, 1);
        check(1, "Apply runs without crash");

        wubu_mmrope_close(ctx);
    }

    /* --- Test 2: head_dim not divisible by 3 --- */
    {
        int pos[1] = {0};
        wubu_mmrope_t *ctx = wubu_mmrope_init(10, 10000.0f, 10000.0f, 10000.0f,
                                              1, 1, 1, pos, pos, pos);
        check(ctx == NULL, "Reject head_dim=10 (not div by 3)");
    }

    /* --- Test 3: head_dim not even --- */
    {
        int pos[1] = {0};
        wubu_mmrope_t *ctx = wubu_mmrope_init(9, 10000.0f, 10000.0f, 10000.0f,
                                              1, 1, 1, pos, pos, pos);
        check(ctx == NULL, "Reject head_dim=9 (odd)");
    }

    /* --- Test 4: Zero position = identity --- */
    {
        int pos_t[1] = {0};
        int pos_h[1] = {0};
        int pos_w[1] = {0};
        wubu_mmrope_t *ctx = wubu_mmrope_init(12, 10000.0f, 10000.0f, 10000.0f,
                                              1, 1, 1, pos_t, pos_h, pos_w);

        float qk[12];
        for (int i = 0; i < 12; i++) qk[i] = (float)(i + 1);
        float orig[12];
        memcpy(orig, qk, 48);

        wubu_mmrope_apply(ctx, qk, 1, 1);

        int identity = 1;
        for (int i = 0; i < 12; i++) {
            if (fabs(qk[i] - orig[i]) > 1e-5f) identity = 0;
        }
        check(identity, "Zero position = identity (no rotation)");
        wubu_mmrope_close(ctx);
    }

    /* --- Test 5: Non-zero position = rotation matches reference --- */
    {
        int pos_t[1] = {5};
        int pos_h[1] = {3};
        int pos_w[1] = {2};
        wubu_mmrope_t *ctx = wubu_mmrope_init(6, 10000.0f, 10000.0f, 10000.0f,
                                              6, 4, 3, pos_t, pos_h, pos_w);

        float qk[6];
        for (int i = 0; i < 6; i++) qk[i] = (float)(i + 1);

        /* Reference: manually apply 3D RoPE */
        float ref[6];
        memcpy(ref, qk, 24);
        /* head_dim=6, seg_dim=2, half_seg=1 */
        /* Segment 0 (temporal): dim 0,1, pos_t=5 */
        ref_rotate(&ref[0], &ref[1], 5.0f, 10000.0f, 2, 0);
        /* Segment 1 (height): dim 2, pos_h=3 */
        ref_rotate(&ref[2], &ref[3], 3.0f, 10000.0f, 2, 0);
        /* Segment 2 (width): dim 4, pos_w=2 */
        ref_rotate(&ref[4], &ref[5], 2.0f, 10000.0f, 2, 0);

        wubu_mmrope_apply(ctx, qk, 1, 1);

        int match = 1;
        for (int i = 0; i < 6; i++) {
            if (fabs(qk[i] - ref[i]) > 1e-5f) match = 0;
        }
        check(match, "Non-zero position matches reference rotation");
        wubu_mmrope_close(ctx);
    }

    /* --- Test 6: NULL safety --- */
    {
        wubu_mmrope_apply(NULL, (float*)1, 1, 1);
        check(1, "NULL ctx does not crash");
        wubu_mmrope_apply((wubu_mmrope_t*)1, NULL, 1, 1);
        check(1, "NULL qk does not crash");
        wubu_mmrope_apply((wubu_mmrope_t*)1, (float*)1, 0, 1);
        check(1, "seq_len=0 does not crash");
    }

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           tests_pass, tests_run, tests_fail);
    return tests_fail > 0 ? 1 : 0;
}
