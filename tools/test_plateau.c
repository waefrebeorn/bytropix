/* test_plateau.c -- the amoeba plateau trigger: a decreasing loss series
 * is progress, a flat/rising one is a plateau (the growth fires). */
#include <stdio.h>
#include "wubu_plateau.h"

int main(void)
{
    int ok = 1;
    /* improving: 10 -> 2 over 20 steps, slope ~ -0.42/step */
    float improving[20];
    for (int i = 0; i < 20; i++) improving[i] = 10.0f - 0.4f * i;
    /* flat: 3.0 with a tiny wobble */
    float flat[20];
    for (int i = 0; i < 20; i++) flat[i] = 3.0f + (i % 3 == 0 ? 0.01f : 0.0f);
    /* rising: diverging */
    float rising[20];
    for (int i = 0; i < 20; i++) rising[i] = 2.0f + 0.1f * i;

    float s1 = wubu_plateau_slope(improving, 20, 10);
    float s2 = wubu_plateau_slope(flat, 20, 10);
    float s3 = wubu_plateau_slope(rising, 20, 10);
    if (s1 > -0.3f || s1 > 0) { printf("  improving slope %.4f FAIL\n", s1); ok = 0; }
    if (s2 > 1e-3f || s2 < -1e-3f) { printf("  flat slope %.4f FAIL\n", s2); ok = 0; }
    if (s3 < 0.05f) { printf("  rising slope %.4f FAIL\n", s3); ok = 0; }

    int d1 = wubu_plateau_detect(improving, 20, 10, 0.01f);
    int d2 = wubu_plateau_detect(flat, 20, 10, 0.01f);
    int d3 = wubu_plateau_detect(rising, 20, 10, 0.01f);
    if (d1 != 0) { printf("  improving should NOT trigger (got %d) FAIL\n", d1); ok = 0; }
    if (d2 != 1) { printf("  flat SHOULD trigger (got %d) FAIL\n", d2); ok = 0; }
    if (d3 != 1) { printf("  rising SHOULD trigger (got %d) FAIL\n", d3); ok = 0; }

    /* the short-history guard: not enough samples -> no trigger */
    if (wubu_plateau_detect(improving, 5, 10, 0.01f) != 0) {
        printf("  short history should not trigger FAIL\n"); ok = 0;
    }
    printf("  slopes %.4f / %.4f / %.4f, triggers %d/%d/%d  %s\n",
           s1, s2, s3, d1, d2, d3, ok ? "PASS" : "FAIL");
    printf("%s\n", ok ? "ALL PLATEAU TESTS PASSED" : "PLATEAU FAILURES");
    return ok ? 0 : 1;
}
