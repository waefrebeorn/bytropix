/* test_avx2_physics.c — WORLDSIM #607: AVX2 path beats scalar on measured bodies.
 *
 * Measured assertions:
 *   1. correctness: AVX2 integration == scalar integration (within tol),
 *      for n=1000 bodies including a non-multiple-of-8 tail.
 *   2. speed: on a large batch (n=1<<20) repeated R times, the AVX2 path's
 *      wall-clock time is < the scalar path's (the measured speedup).
 *      (When __AVX2__ is absent, both paths are scalar; the test then
 *      asserts only correctness and reports the build has no AVX2.)
 */
#include "wubu_simd_physics.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static int g_run = 0, g_pass = 0;
#define T(cond, msg) do { \
    g_run++; \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while (0)

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void fill(float *a, int n, unsigned seed) {
    for (int i = 0; i < n; i++) {
        seed = seed * 1664525u + 1013904223u;
        a[i] = ((float)(seed % 20000) / 10000.0f - 1.0f) * 5.0f;
    }
}

int main(void) {
    printf("=== test_avx2_physics (vectorized physics, #607) ===\n");
    printf("  build AVX2: %s\n", wubu_simd_has_avx2() ? "yes" : "no (scalar fallback)");

    /* 1: correctness over n=1000 (not a multiple of 8 -> tail path). */
    {
        const int n = 1000;
        float *dx = malloc(n*4), *dy = malloc(n*4), *dz = malloc(n*4);
        float *pax = malloc(n*4), *pay = malloc(n*4), *paz = malloc(n*4);
        float *pbx = malloc(n*4), *pby = malloc(n*4), *pbz = malloc(n*4);
        fill(dx, n, 1u); fill(dy, n, 2u); fill(dz, n, 3u);
        fill(pax, n, 4u); fill(pay, n, 5u); fill(paz, n, 6u);
        for (int i = 0; i < n; i++) { pbx[i]=pax[i]; pby[i]=pay[i]; pbz[i]=paz[i]; }

        wubu_simd_integrate_scalar(dx,dy,dz, pax,pay,paz, n, 0.01f, -9.81f, 1.0f/60.0f);
        wubu_simd_integrate_avx2  (dx,dy,dz, pbx,pby,pbz, n, 0.01f, -9.81f, 1.0f/60.0f);

        double maxd = 0.0;
        for (int i = 0; i < n; i++) {
            double d;
            d = fabs((double)pax[i]-pbx[i]); if (d>maxd) maxd=d;
            d = fabs((double)pay[i]-pby[i]); if (d>maxd) maxd=d;
            d = fabs((double)paz[i]-pbz[i]); if (d>maxd) maxd=d;
        }
        printf("  max |scalar-avx2| = %.3e\n", maxd);
        T(maxd < 1e-4, "AVX2 == scalar within 1e-4 (incl. tail)");
        free(dx);free(dy);free(dz);free(pax);free(pay);free(paz);free(pbx);free(pby);free(pbz);
    }

    /* 2: speed on a large batch. */
    if (wubu_simd_has_avx2()) {
        const int n = 1 << 20;    /* 1,048,576 bodies */
        const int R = 20;         /* repeats to get a measurable time */
        float *dx = malloc(n*4), *dy = malloc(n*4), *dz = malloc(n*4);
        float *px = malloc(n*4), *py = malloc(n*4), *pz = malloc(n*4);
        fill(dx, n, 7u); fill(dy, n, 8u); fill(dz, n, 9u);
        fill(px, n, 10u); fill(py, n, 11u); fill(pz, n, 12u);

        double t0 = now_sec();
        for (int r = 0; r < R; r++)
            wubu_simd_integrate_scalar(dx,dy,dz, px,py,pz, n, 0.01f, -9.81f, 1.0f/60.0f);
        double t_scalar = now_sec() - t0;

        fill(px, n, 10u); fill(py, n, 11u); fill(pz, n, 12u);
        t0 = now_sec();
        for (int r = 0; r < R; r++)
            wubu_simd_integrate_avx2(dx,dy,dz, px,py,pz, n, 0.01f, -9.81f, 1.0f/60.0f);
        double t_avx2 = now_sec() - t0;

        double speedup = t_avx2 > 0 ? t_scalar / t_avx2 : 0.0;
        printf("  bodies=%d reps=%d  scalar=%.3fs  avx2=%.3fs  speedup=%.2fx\n",
               n, R, t_scalar, t_avx2, speedup);
        T(t_avx2 < t_scalar, "AVX2 wall-clock < scalar on measured bodies");
        T(speedup > 1.0, "measured speedup > 1.0x");
        free(dx);free(dy);free(dz);free(px);free(py);free(pz);
    } else {
        printf("  (no AVX2 in this build: speed assertion skipped, correctness only)\n");
    }

    printf("=== Results: %d/%d passed ===\n", g_pass, g_run);
    return g_pass == g_run ? 0 : 1;
}
