/* test_recency.c -- the recency-weighted sampling: monotonic increase
 * toward the fresh end, the base floor, the power curve. */
#include <stdio.h>
#include <math.h>
#include "wubu_recency.h"

int main(void)
{
    int ok = 1;
    long n = 1000;
    float w0 = wubu_recency_weight(0, n, 0.2f, 1.0f);
    float w500 = wubu_recency_weight(500, n, 0.2f, 1.0f);
    float w999 = wubu_recency_weight(999, n, 0.2f, 1.0f);
    if (fabsf(w0 - 0.2f) > 1e-4f) { printf("  base %.4f FAIL\n", w0); ok = 0; }
    if (fabsf(w999 - 1.0f) > 1e-3f) { printf("  fresh %.4f FAIL\n", w999); ok = 0; }
    if (!(w500 > w0 && w999 > w500)) { printf("  monotonic FAIL\n"); ok = 0; }
    /* the linear midpoint: 0.2 + 0.8*0.5 = 0.6 */
    if (fabsf(w500 - 0.6f) > 1e-3f) { printf("  midpoint %.4f FAIL\n", w500); ok = 0; }

    /* the power curve: power 2 pulls the fresh end harder */
    float p0 = wubu_recency_weight(0, n, 0.2f, 2.0f);
    float p500 = wubu_recency_weight(500, n, 0.2f, 2.0f);
    if (fabsf(p0 - 0.2f) > 1e-4f) { printf("  power base FAIL\n"); ok = 0; }
    if (!(p500 < 0.6f - 1e-3f)) { printf("  power mid %.4f < linear FAIL\n", p500); ok = 0; }

    /* the freshness ratio: the last decile vs the first decile */
    float last = wubu_recency_weight(990, n, 0.2f, 1.0f);
    float first = wubu_recency_weight(10, n, 0.2f, 1.0f);
    if (last / first < 3.0f) { printf("  fresh ratio %.2f FAIL\n", last / first); ok = 0; }

    printf("  recency: w0=%.3f w500=%.3f w999=%.3f power-mid=%.3f ratio=%.2f  %s\n",
           w0, w500, w999, p500, last / first, ok ? "PASS" : "FAIL");
    printf("%s\n", ok ? "ALL RECENCY TESTS PASSED" : "RECENCY FAILURES");
    return ok ? 0 : 1;
}
