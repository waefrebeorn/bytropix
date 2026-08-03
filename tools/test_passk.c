/* test_passk.c -- the pass@k estimator: the unbiased combinatorial
 * values against the hand-computed references. */
#include <stdio.h>
#include <math.h>
#include "wubu_passk.h"

int main(void)
{
    int ok = 1;
    /* 10 attempts, 5 successes: pass@1 = 0.5; pass@2 = 1 - C(5,2)/C(10,2)
     * = 1 - 10/45 = 0.7778; pass@3 = 1 - C(5,3)/C(10,3) = 1 - 10/120 = 0.9167 */
    int succ[10];
    for (int i = 0; i < 10; i++) succ[i] = (i < 5) ? 1 : 0;
    float p1 = wubu_passk(succ, 10, 1);
    float p2 = wubu_passk(succ, 10, 2);
    float p3 = wubu_passk(succ, 10, 3);
    if (fabsf(p1 - 0.5f) > 1e-4f) { printf("  pass@1 %.4f FAIL\n", p1); ok = 0; }
    if (fabsf(p2 - 0.7778f) > 1e-3f) { printf("  pass@2 %.4f FAIL\n", p2); ok = 0; }
    if (fabsf(p3 - 0.9167f) > 1e-3f) { printf("  pass@3 %.4f FAIL\n", p3); ok = 0; }

    /* the edge cases: all-fail -> 0, all-succeed -> 1, k=n -> 1 */
    int fail[5] = {0, 0, 0, 0, 0};
    int all[5] = {1, 1, 1, 1, 1};
    if (wubu_passk(fail, 5, 2) != 0) { printf("  all-fail FAIL\n"); ok = 0; }
    if (wubu_passk(all, 5, 2) != 1) { printf("  all-succeed FAIL\n"); ok = 0; }
    if (wubu_passk(succ, 10, 10) != 1) { printf("  k=n FAIL\n"); ok = 0; }
    if (wubu_passk(succ, 10, 11) != 0) { printf("  k>n FAIL\n"); ok = 0; }

    /* monotonicity: pass@k grows with k */
    if (!(p3 > p2 && p2 > p1)) { printf("  monotonicity FAIL\n"); ok = 0; }

    printf("  pass@1 %.4f pass@2 %.4f pass@3 %.4f  %s\n", p1, p2, p3,
           ok ? "PASS" : "FAIL");
    printf("%s\n", ok ? "ALL PASSK TESTS PASSED" : "PASSK FAILURES");
    return ok ? 0 : 1;
}
