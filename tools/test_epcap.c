/* test_epcap.c -- the episode-length cap: the longest fitting prefix,
 * the full-fits flag, the zero-cost steps. */
#include <stdio.h>
#include "wubu_epcap.h"

int main(void)
{
    int ok = 1;
    /* costs 10,20,30,40 with a 60-budget: keep 3 (sum 60) */
    int cost[] = { 10, 20, 30, 40 };
    int kept = 0;
    int full = wubu_epcap(cost, 4, 60, &kept);
    if (kept != 3 || full != 0) {
        printf("  cap 60 -> kept %d full %d FAIL\n", kept, full); ok = 0;
    }
    /* a 100-budget fits everything */
    full = wubu_epcap(cost, 4, 100, &kept);
    if (kept != 4 || full != 1) {
        printf("  cap 100 -> kept %d full %d FAIL\n", kept, full); ok = 0;
    }
    /* zero-cost steps (user turns) are always kept */
    int cost2[] = { 0, 50, 0, 50 };
    full = wubu_epcap(cost2, 4, 60, &kept);
    if (kept != 3) { printf("  zero-cost kept %d FAIL\n", kept); ok = 0; }
    /* a single over-budget step truncates to 0 */
    int cost3[] = { 70 };
    full = wubu_epcap(cost3, 1, 60, &kept);
    if (kept != 0 || full != 0) { printf("  over-budget FAIL\n"); ok = 0; }
    int k2 = 0; wubu_epcap(cost2, 4, 60, &k2);
    printf("  epcap: 60->3 full-ok 100->4 zero-cost->3 over->0  %s\n",
           ok ? "PASS" : "FAIL");
    printf("%s\n", ok ? "ALL EPCAP TESTS PASSED" : "EPCAP FAILURES");
    return ok ? 0 : 1;
}
