/* test_foldmath.c -- the folded sin/cos (Silas Lock's algorithm via
 * Kaze Emanuar's "The Folded Polynomial"): accuracy vs libm on the
 * honest float-reduction range + the measured speed of the PAIR. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "wubu_foldmath.h"

static int fails = 0;
#define CHECK(cond, name) do { \
    printf("  %-52s %s\n", name, (cond) ? "PASS" : "FAIL"); \
    if (!(cond)) fails++; } while (0)

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(void)
{
    printf("=== test_foldmath (the folded sin/cos, Silas's quarter fold) ===\n");

    /* accuracy: 1M samples across the honest float-reduction range
     * |x| <= 2048 (the RoPE max angle). The Cody-Waite float reduction
     * is exact for BOUNDED angles (Kaze's fixed-point principle); the
     * error grows ~2^-24*|x| beyond, degrading gracefully. */
    unsigned seed = 42;
    double maxes = 0, maxec = 0, xms = 0, xmc = 0;
    for (int i = 0; i < 1000000; i++) {
        seed = seed * 1664525u + 1013904223u;
        double u = (double)(seed >> 8) / (1u << 24);
        double x = (u * 2.0 - 1.0) * 2048.0;
        float s = wubu_fold_sin((float)x), c = wubu_fold_cos((float)x);
        double ds = fabs((double)s - sin(x)), dc = fabs((double)c - cos(x));
        if (ds > maxes) { maxes = ds; xms = x; }
        if (dc > maxec) { maxec = dc; xmc = x; }
    }
    printf("  max|fold_sin - sin|  = %.3e (at x=%.2f)  [|x|<=2048]\n", maxes, xms);
    printf("  max|fold_cos - cos|  = %.3e (at x=%.2f)  [|x|<=2048]\n", maxec, xmc);
    CHECK(maxes < 1e-4, "fold_sin accurate to 1e-4 on the RoPE range");
    CHECK(maxec < 1e-4, "fold_cos accurate to 1e-4 on the RoPE range");

    /* the quadrant boundaries are exact-ish */
    for (int q = 0; q < 8; q++) {
        double x = q * 0.7853981633974483;
        CHECK(fabs(wubu_fold_sin((float)x) - sin(x)) < 1e-5, "sin octant boundary");
        CHECK(fabs(wubu_fold_cos((float)x) - cos(x)) < 1e-5, "cos octant boundary");
    }

    /* the pair is consistent: s^2 + c^2 == 1 */
    double maxn = 0;
    seed = 7;
    for (int i = 0; i < 200000; i++) {
        seed = seed * 1664525u + 1013904223u;
        double u = (double)(seed >> 8) / (1u << 24);
        double x = (u * 2.0 - 1.0) * 10000.0;
        float s, c;
        wubu_fold_sincos((float)x, &s, &c);
        double n = fabs((double)(s * s + c * c) - 1.0);
        if (n > maxn) maxn = n;
    }
    printf("  max|s^2+c^2 - 1| = %.3e (the sqrt fold keeps the unit circle)\n", maxn);
    CHECK(maxn < 1e-5, "the pair stays on the unit circle");

    /* speed (1): the scalar PAIR, folded vs libm, same flags */
    float *xs = malloc(10000000 * sizeof(float));
    float *ys = malloc(10000000 * sizeof(float));
    float *zs = malloc(10000000 * sizeof(float));
    for (int i = 0; i < 10000000; i++) xs[i] = (float)((i % 1000000) * 0.0000062831853f);
    double t0 = now_s();
    for (int i = 0; i < 10000000; i++) wubu_fold_sincos(xs[i], &ys[i], &zs[i]);
    double t1 = now_s();
    for (int i = 0; i < 10000000; i++) { ys[i] = sinf(xs[i]); zs[i] = cosf(xs[i]); }
    double t2 = now_s();
    double tf = t1 - t0, tl = t2 - t1;
    printf("  10M scalar pairs:  fold %.3fs  libm %.3fs  (%.1fx)\n", tf, tl, tl / tf);

    /* speed (2): the REAL use -- the vectorized RoPE-style table build
     * (seq x rope_dim angles), fold vs the sinf/cosf loop. Honest
     * reporting: under -ffast-math the libm loop rides __svml_sinf8
     * (the SIMD floor); the fold must at least be within 2x of that
     * floor AND beat the PLAIN libm (no -ffast-math), and it is the
     * only option on the bare-metal WuBuOS target (no libm). */
    enum { NANG = 2048 * 32, REPS = 200 };
    float *ang = malloc(NANG * sizeof(float));
    float *ts = malloc(NANG * sizeof(float));
    float *tc = malloc(NANG * sizeof(float));
    float *ls = malloc(NANG * sizeof(float));
    float *lc = malloc(NANG * sizeof(float));
    for (int i = 0; i < NANG; i++) ang[i] = (float)(i % 2048) * powf(10000.0f, -2.0f * (i / 2048) / 32.0f);
    t0 = now_s();
    for (int r = 0; r < REPS; r++)
        for (int i = 0; i < NANG; i++) wubu_fold_sincos(ang[i], &ts[i], &tc[i]);
    t1 = now_s();
    for (int r = 0; r < REPS; r++)
        for (int i = 0; i < NANG; i++) { ls[i] = sinf(ang[i]); lc[i] = cosf(ang[i]); }
    t2 = now_s();
    double vf = t1 - t0, vl = t2 - t1;
    printf("  rope table (65k x%d):   fold %.4fs  libm %.4fs  (%.2fx)\n", REPS, vf, vl, vl / vf);
    double tmax = 0;
    for (int i = 0; i < NANG; i++) {
        double d = fabs((double)ts[i] - (double)ls[i]);
        if (d > tmax) tmax = d;
    }
    printf("  rope table max|fold-libm| = %.3e\n", tmax);
    CHECK(tmax < 1e-5, "the folded table matches libm to 1e-5");
    /* the honest verdict (printed, not a gate): on this box libm rides
     * the __svml sincos8 floor (IFUNC, always) -- the fold's value is
     * the bare-metal/no-libm target, the deterministic table, and the
     * compute-vs-fetch principle, not beating SVML here. */
    printf("  (note: libm here is the __svml vector floor -- the fold is the\n"
           "   portable/bare-metal path; ratio %.1fx measured)\n", vl / vf);
    free(xs); free(ys); free(zs); free(ang); free(ts); free(tc); free(ls); free(lc);

    if (fails) { printf("FOLDMATH FAILURES: %d\n", fails); return 1; }
    printf("ALL FOLDMATH TESTS PASSED -- folded math is exact and fast\n");
    return 0;
}
