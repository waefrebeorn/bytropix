/*
 * test_wubu_mhc_mh.c -- multi-head hyper-connections (2512.24880 form).
 *
 * DA oracles:
 *   1. identity init: read(h, i) returns h[i] exactly (maxdiff < 1e-6)
 *   2. function-preserving write: alpha=1.0 leaves h[i] unchanged (< 1e-6)
 *   3. random M (constrained): read == hand-computed sum_k M[i,k]*h[k]
 *      (< 1e-4)
 *   4. manifold: every row of M sums to 1.0 within 1e-4 and stays in [0,1]
 *   5. round-trip read/write keeps h finite (no NaN), nh=4 d=64
 */
#include "wubu_mhc_mh.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } \
                           else { printf("  ok: %s\n", msg); } } while (0)

static double maxdiff(const float *a, const float *b, int n)
{
    double d = 0;
    for (int i = 0; i < n; i++) {
        double x = fabs((double)a[i] - (double)b[i]);
        if (x > d) d = x;
    }
    return d;
}

int main(void)
{
    printf("=== test_wubu_mhc_mh (multi-head hyper-connections) ===\n");
    const int NH = 4, D = 64;
    wubu_mhc_mh_t *m = wubu_mhc_mh_create(NH, D, 48);
    CHECK(m != NULL, "create nh=4 d=64");
    CHECK(m && wubu_mhc_mh_nh(m) == NH && wubu_mhc_mh_dim(m) == D,
          "dims reported");

    float *h[NH];
    for (int i = 0; i < NH; i++) h[i] = (float *)malloc(sizeof(float) * D);
    for (int i = 0; i < NH; i++)
        for (int j = 0; j < D; j++) h[i][j] = (float)((i + 1) * 0.5f + j * 0.01f);

    /* 1. identity read */
    wubu_mhc_mh_set_identity(m);
    CHECK(m && wubu_mhc_mh_manifold_ok(m), "identity is a valid manifold");
    {
        float out[D];
        float err = 0;
        for (int i = 0; i < NH; i++) {
            wubu_mhc_mh_read(m, h, i, out);
            double d = maxdiff(out, h[i], D);
            if (d > err) err = d;
        }
        printf("  identity read maxdiff = %g\n", err);
        CHECK(err < 1e-6, "identity read == h[i] exactly");
    }

    /* 2. function-preserving write */
    {
        float copy[NH][D];
        for (int i = 0; i < NH; i++) memcpy(copy[i], h[i], sizeof(float) * D);
        float y[D];
        for (int j = 0; j < D; j++) y[j] = 1.0f;
        for (int i = 0; i < NH; i++) wubu_mhc_mh_write(m, h[i], y, 1.0f);
        double err = 0;
        for (int i = 0; i < NH; i++) {
            double d = maxdiff(h[i], copy[i], D);
            if (d > err) err = d;
        }
        printf("  alpha=1 write maxdiff = %g\n", err);
        CHECK(err < 1e-6, "alpha=1.0 write leaves h unchanged (residual)");
    }

    /* 3. random constrained M: read == hand-computed combo */
    {
        wubu_mhc_mh_t *r = wubu_mhc_mh_create(NH, D, 7);
        CHECK(r && wubu_mhc_mh_manifold_ok(r), "random M constrained to simplex");
        float out[D];
        double err = 0;
        for (int i = 0; i < NH; i++) {
            wubu_mhc_mh_read(r, h, i, out);
            const float *row = wubu_mhc_mh_mixing_row(r, i);
            for (int j = 0; j < D; j++) {
                float want = 0;
                for (int k = 0; k < NH; k++) want += row[k] * h[k][j];
                double d = fabs((double)out[j] - (double)want);
                if (d > err) err = d;
            }
        }
        printf("  random-M read vs hand-computed maxdiff = %g\n", err);
        CHECK(err < 1e-4, "read == sum_k M[i,k]*h[k] (< 1e-4)");
        wubu_mhc_mh_free(r);
    }

    /* 4. manifold property */
    CHECK(wubu_mhc_mh_manifold_ok(m), "identity rows sum to 1 in [0,1]");
    {
        wubu_mhc_mh_t *r = wubu_mhc_mh_create(NH, D, 99);
        CHECK(r && wubu_mhc_mh_manifold_ok(r), "random rows sum to 1 in [0,1]");
        wubu_mhc_mh_free(r);
    }

    /* 5. round-trip read/write stays finite (no NaN) for many steps */
    {
        int finite = 1;
        float y[D];
        for (int step = 0; step < 200; step++) {
            for (int j = 0; j < D; j++) y[j] = sinf((float)step * 0.1f) * 2.0f;
            for (int i = 0; i < NH; i++) {
                wubu_mhc_mh_write(m, h[i], y, 0.9f);
                float out[D];
                wubu_mhc_mh_read(m, h, i, out);
                for (int j = 0; j < D; j++)
                    if (!isfinite(out[j]) || !isfinite(h[i][j])) finite = 0;
            }
        }
        CHECK(finite, "200-step read/write round-trip stays finite");
    }

    /* edge: create rejects bad args */
    CHECK(wubu_mhc_mh_create(0, D, 1) == NULL, "nh=0 rejected");
    CHECK(wubu_mhc_mh_create(NH, 0, 1) == NULL, "d=0 rejected");
    CHECK(wubu_mhc_mh_read(NULL, h, 0, NULL) == -1, "null read rejected");
    CHECK(wubu_mhc_mh_mixing_row(m, 99) == NULL, "oob row rejected");

    for (int i = 0; i < NH; i++) free(h[i]);
    wubu_mhc_mh_free(m);

    if (failures == 0) { printf("ALL WUBU_MHC_MH TESTS PASSED\n"); return 0; }
    printf("%d FAILURES\n", failures);
    return 1;
}
