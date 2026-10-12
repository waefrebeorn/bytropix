/* test_data_mix.c — BRAIN #243: data mix forward verified.
 *
 * Measured assertions on wubu_data_mix + wubu_data_mix_sample:
 *   1. proportions sum to 1.
 *   2. exact normalization (70/20/10 -> 0.7/0.2/0.1).
 *   3. equal weights -> uniform 1/k.
 *   4. sampling hits the right bucket by cumulative boundary.
 *   5. sampling frequency ~ proportions over many draws (statistical).
 *   6. bad args rejected (negative, all-zero, NULL, k<=0).
 */
#include "wubu_data_mix.h"
#include <stdio.h>
#include <math.h>

static int g_run = 0, g_pass = 0;
#define T(cond, msg) do { \
    g_run++; \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while (0)

static int near(double a, double b, double eps) { return fabs(a - b) <= eps; }

int main(void) {
    printf("=== test_data_mix (data mixture weighting, #243) ===\n");

    /* 1+2: sum to 1, exact 70/20/10. */
    double w[3] = {70, 20, 10};
    double p[3];
    T(wubu_data_mix(w, 3, p) == 0, "mix ok");
    double sum = p[0]+p[1]+p[2];
    T(near(sum, 1.0, 1e-9), "proportions sum to 1");
    T(near(p[0],0.7,1e-9) && near(p[1],0.2,1e-9) && near(p[2],0.1,1e-9),
      "exact 70/20/10 -> 0.7/0.2/0.1");

    /* 3: equal weights -> uniform. */
    double we[4] = {1,1,1,1};
    double pe[4];
    wubu_data_mix(we, 4, pe);
    int uni = 1; for (int i=0;i<4;i++) if (!near(pe[i], 0.25, 1e-9)) uni = 0;
    T(uni, "equal weights -> uniform 1/k");

    /* 4: bucket by cumulative boundary. p = 0.7/0.2/0.1. */
    T(wubu_data_mix_sample(p, 3, 0.0) == 0, "u=0 -> bucket 0");
    T(wubu_data_mix_sample(p, 3, 0.69) == 0, "u=0.69 -> bucket 0");
    T(wubu_data_mix_sample(p, 3, 0.71) == 1, "u=0.71 -> bucket 1");
    T(wubu_data_mix_sample(p, 3, 0.91) == 2, "u=0.91 -> bucket 2");
    T(wubu_data_mix_sample(p, 3, 0.999) == 2, "u~1 -> last bucket");

    /* 5: frequency ~ proportions over many draws. */
    const int N = 100000;
    int cnt[3] = {0,0,0};
    for (int i = 0; i < N; i++) {
        double u = (double)i / (double)N;   /* deterministic sweep */
        cnt[wubu_data_mix_sample(p, 3, u)]++;
    }
    T(near((double)cnt[0]/N, 0.7, 0.01) &&
      near((double)cnt[1]/N, 0.2, 0.01) &&
      near((double)cnt[2]/N, 0.1, 0.01), "sampling frequency ~ proportions");

    /* 6: bad args. */
    T(wubu_data_mix(NULL, 3, p) == -1, "NULL weights rejected");
    T(wubu_data_mix(w, 0, p) == -1, "k<=0 rejected");
    double wneg[2] = {1, -1};
    T(wubu_data_mix(wneg, 2, p) == -1, "negative weight rejected");
    double wz[2] = {0, 0};
    T(wubu_data_mix(wz, 2, p) == -1, "all-zero rejected");
    T(wubu_data_mix_sample(p, 3, 1.5) == -1, "u out of range rejected");

    printf("=== Results: %d/%d passed ===\n", g_pass, g_run);
    return g_pass == g_run ? 0 : 1;
}
