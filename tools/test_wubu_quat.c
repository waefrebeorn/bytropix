/*
 * test_wubu_quat.c -- Known-Answer Tests for wubu_quat.h.
 *
 * Expected vectors are taken from the independent Python reference
 * quat_oracle.py (scalar-first w,x,y,z), which is cross-checked against
 * analytic identities before hardcoding:
 *   - PI * PI == (-1,0,0,0)   (two 180deg rotations = full turn, q double cover)
 *   - slerp(I, PI, 0.5) == 90deg about x == (1/sqrt2, 1/sqrt2, 0, 0)
 *   - slerp shortest-path flip: slerp(I, -I, 0.5) -> identity
 *   - 360deg about z -> -I (double cover), 90deg about z maps x->y
 *
 * C floats vs the double oracle differ by ~1e-7; the KAT tolerance is 1e-5f,
 * chosen to absorb float rounding while still catching formula drift.
 */
#include <stdio.h>
#include <math.h>
#include "wubu_quat.h"

static unsigned g_pass = 0, g_fail = 0;
#define EPS 1e-5f

/* compare two quaternions component-wise; prints expect/got on mismatch so a
 * wrong RESULT (not just a wrong sign) is visible for diagnosis. */
static int quat_close(const char *name, const char *file, int line,
                      wubu_quat got, float ew, float ex, float ey, float ez)
{
    wubu_quat want = { ew, ex, ey, ez };
    int ok = (fabsf(got.w - want.w) <= EPS)
          && (fabsf(got.x - want.x) <= EPS)
          && (fabsf(got.y - want.y) <= EPS)
          && (fabsf(got.z - want.z) <= EPS);
    if (!ok) {
        printf("FAIL: %s (%s:%d)\n  expect (%.9f, %.9f, %.9f, %.9f)\n"
               "   got (%.9f, %.9f, %.9f, %.9f)\n",
               name, file, line, want.w, want.x, want.y, want.z,
               got.w, got.x, got.y, got.z);
        g_fail++;
    } else {
        g_pass++;
    }
    return ok;
}
#define CHECKQ(name, got, W, X, Y, Z) \
    quat_close(name, __FILE__, __LINE__, got, (W), (X), (Y), (Z))

static int vec_close(const char *name, const char *file, int line,
                     const float got[3], float ex, float ey, float ez)
{
    int ok = fabsf(got[0] - ex) <= EPS && fabsf(got[1] - ey) <= EPS
          && fabsf(got[2] - ez) <= EPS;
    if (!ok) {
        printf("FAIL: %s (%s:%d)\n  expect (%.9f, %.9f, %.9f)\n   got (%.9f, %.9f, %.9f)\n",
               name, file, line, ex, ey, ez, got[0], got[1], got[2]);
        g_fail++;
    } else {
        g_pass++;
    }
    return ok;
}
#define CHECKV(name, got, X, Y, Z) \
    vec_close(name, __FILE__, __LINE__, got, (X), (Y), (Z))

int main(void)
{
    const wubu_quat I  = { 1.0f, 0.0f, 0.0f, 0.0f };
    const wubu_quat PI = { 0.0f, 1.0f, 0.0f, 0.0f };  /* 180deg about x */
    const wubu_quat Z  = { 0.0f, 0.0f, 0.0f, 0.0f };
    const wubu_quat QZ90 = { 0.7071067812f, 0.0f, 0.0f, 0.7071067812f }; /* 90deg/z */
    const wubu_quat mPI = { -1.0f, 0.0f, 0.0f, 0.0f }; /* -I */

    /* construction + algebra ------------------------------------------------ */
    CHECKQ("identity",            wubu_quat_identity(), 1.0f, 0.0f, 0.0f, 0.0f);
    CHECKQ("two180_mul",         wubu_quat_mul(PI, PI), -1.0f, 0.0f, 0.0f, 0.0f);
    CHECKQ("norm_zero",          wubu_quat_normalize(Z), 1.0f, 0.0f, 0.0f, 0.0f);
    CHECKQ("conjugate_pi",       wubu_quat_conjugate(PI),  0.0f,-1.0f, 0.0f, 0.0f);
    CHECKQ("inverse_pi",         wubu_quat_inverse(PI),    0.0f,-1.0f, 0.0f, 0.0f);
    CHECKQ("add_identity_pi",    wubu_quat_add(I, PI),     1.0f, 1.0f, 0.0f, 0.0f);
    CHECKQ("axisangle_180x",
            wubu_quat_from_axis_angle((float[3]){1,0,0}, (float)M_PI),
            0.0f, 1.0f, 0.0f, 0.0f);                    /* 180deg about x -> (0,1,0,0) */
    CHECKQ("axisangle_unit_z90",
            wubu_quat_from_axis_angle((float[3]){0,0,1}, (float)(M_PI/2)),
            0.7071067812f, 0.0f, 0.0f, 0.7071067812f);
    CHECKQ("degenerate_axis",
            wubu_quat_from_axis_angle((float[3]){0,0,0}, 1.0f),
            1.0f, 0.0f, 0.0f, 0.0f);

    /* rotation ------------------------------------------------------------- */
    {
        float r[3];
        wubu_quat_rotate(QZ90, (float[3]){1,0,0}, r);
        CHECKV("rotate_x_by_z90", r, 0.0f, 1.0f, 0.0f);   /* +90deg/z: x -> y */
        wubu_quat_rotate(QZ90, (float[3]){0,1,0}, r);
        CHECKV("rotate_y_by_z90", r, -1.0f, 0.0f, 0.0f);  /* +90deg/z: y -> -x */
    }

    /* slerp ---------------------------------------------------------------- */
    CHECKQ("slerp_mid",          wubu_quat_slerp(I, PI, 0.5f),
            0.7071067812f, 0.7071067812f, 0.0f, 0.0f);   /* 90deg about x */
    CHECKQ("slerp_flip",         wubu_quat_slerp(I, (wubu_quat){0,-1,0,0}, 0.5f),
            0.7071067812f, -0.7071067812f, 0.0f, 0.0f);
    CHECKQ("slerp_neg_w",        wubu_quat_slerp(I, mPI, 0.5f), 1.0f,0,0,0); /* flip->I  */
    CHECKQ("slerp_zero_norm_fix",wubu_quat_slerp(I, Z, 0.5f), 1.0f,0,0,0);   /* degenerate -> a */
    CHECKQ("slerp_identity_025", wubu_quat_slerp(I, I, 0.25f), 1.0f,0,0,0);
    CHECKQ("slerp_end0",         wubu_quat_slerp(I, PI, 0.0f), 1.0f,0,0,0);
    CHECKQ("slerp_end1",         wubu_quat_slerp(I, PI, 1.0f), 0.0f,1.0f,0,0);

    /* integrate ----------------------------------------------------------- */
    CHECKQ("integrate_qz90_oz_pi2_dt1",
            wubu_quat_integrate(QZ90, (float[3]){0,0,(float)M_PI/2}, 1.0f),
            0.0f, 0.0f, 0.0f, 1.0f);                     /* 90+90=180 about z */
    CHECKQ("integrate_2pi_z",
            wubu_quat_integrate(I, (float[3]){0,0,(float)(2*M_PI)}, 1.0f),
            -1.0f, 0.0f, 0.0f, 0.0f);                    /* 360deg -> -I (cover) */

    /* squad ---------------------------------------------------------------- */
    {
        wubu_quat s1, s2;
        wubu_quat seq0 = I, seq1 = PI, seq2 = wubu_quat_mul(PI, PI), seq3 = QZ90;
        wubu_quat_squad_tangents(seq0, seq1, seq2, seq3, &s1, &s2);
        CHECKQ("squad_ep0", wubu_quat_squad(seq1, s1, s2, seq2, 0.0f),
                0.0f, 1.0f, 0.0f, 0.0f);                /* == q1 */
        CHECKQ("squad_ep1", wubu_quat_squad(seq1, s1, s2, seq2, 1.0f),
                -1.0f, 0.0f, 0.0f, 0.0f);              /* == q2 */
        CHECKQ("squad_mid", wubu_quat_squad(seq1, s1, s2, seq2, 0.5f),
                -0.1554859f, 0.9659481f, 0.0f, 0.2068052f);
    }

    /* mul associativity of two z90 -> 180 about z --------------------------- */
    CHECKQ("quat_mul_qz90_qz90", wubu_quat_mul(QZ90, QZ90),
            0.0f, 0.0f, 0.0f, 1.0f);

    printf("PASS %u/%u\n", g_pass, g_pass + g_fail);
    if (g_fail) { printf("ALL TESTS FAILED (%u)\n", g_fail); return 1; }
    printf("ALL TESTS PASSED\n");
    return 0;
}
