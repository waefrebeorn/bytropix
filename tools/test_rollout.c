/* test_rollout.c -- the Balanced Adaptive Rollout: the harder tasks get
 * more of the budget, the counts sum EXACTLY to the budget, and the
 * unknown-difficulty tasks sit in the middle. */
#include <stdio.h>
#include <string.h>
#include "wubu_rollout.h"

int main(void)
{
    int ok = 1;
    /* 3 tasks: easy (0.9), medium (0.5), hard (0.1) */
    float succ[] = { 0.9f, 0.5f, 0.1f };
    int out[3];
    int r = wubu_rollout_alloc(succ, 3, 100, 1.0f, out);
    if (!r) { printf("  alloc FAIL\n"); return 1; }
    if (out[0] + out[1] + out[2] != 100) {
        printf("  sum %d != 100 FAIL\n", out[0] + out[1] + out[2]); ok = 0;
    }
    if (!(out[2] > out[1] && out[1] > out[0])) {
        printf("  ordering %d %d %d (want hard > med > easy) FAIL\n",
               out[0], out[1], out[2]); ok = 0;
    }
    /* the expected proportions: fail rates 0.1/0.5/0.9 -> 1:5:9 */
    if (out[0] < 4 || out[0] > 10 || out[1] < 28 || out[1] > 40 ||
        out[2] < 55 || out[2] > 68) {
        printf("  proportions %d %d %d (want ~7/33/60) FAIL\n",
               out[0], out[1], out[2]); ok = 0;
    }

    /* the gamma escalation: gamma=2 leans even harder into the hard task */
    int out2[3];
    wubu_rollout_alloc(succ, 3, 100, 2.0f, out2);
    if (out2[2] <= out[2]) {
        printf("  gamma escalation %d -> %d FAIL\n", out[2], out2[2]); ok = 0;
    }

    /* the unknown-difficulty task: sits between the known */
    float succ2[] = { 0.9f, -1.0f, 0.1f };
    int out3[3];
    wubu_rollout_alloc(succ2, 3, 100, 1.0f, out3);
    if (!(out3[2] > out3[1] && out3[1] > out3[0])) {
        printf("  unknown ordering %d %d %d FAIL\n", out3[0], out3[1], out3[2]); ok = 0;
    }

    printf("  rollout alloc: %d %d %d, gamma2 %d, unknown mid %d %d %d  %s\n",
           out[0], out[1], out[2], out2[2], out3[0], out3[1], out3[2],
           ok ? "PASS" : "FAIL");
    printf("%s\n", ok ? "ALL ROLLOUT TESTS PASSED" : "ROLLOUT FAILURES");
    return ok ? 0 : 1;
}
