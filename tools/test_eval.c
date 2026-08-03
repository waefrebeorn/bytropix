/* test_eval.c -- the agentic eval harness: the DB-state verification
 * composed with the pass@k and the format rate. */
#include <stdio.h>
#include <stdlib.h>
#include "wubu_eval.h"

int main(void)
{
    int ok = 1;
    /* 10 trajectories, each with a {price < 500} goal; 6 succeed */
    wubu_db_goal_t goal = { "price", "<", "500" };
    wubu_db_goal_t goals[10];
    wubu_eval_traj_t trajs[10];
    for (int t = 0; t < 10; t++) {
        goals[t] = goal;
        static wubu_db_slot_t s0 = { "price", "700" };
        static wubu_db_slot_t s1 = { "price", "312" };
        trajs[t].state = (t < 6) ? &s1 : &s0;
        trajs[t].nslots = 1;
        trajs[t].format_ok = (t % 2 == 0) ? 1 : 0;   /* 5 of 10 format-ok */
    }
    int n_ok = 0;
    float pass1 = 0, passk = 0, fmt = 0;
    if (!wubu_eval_run(goals, trajs, 10, 3, &n_ok, &pass1, &passk, &fmt)) {
        printf("  eval FAIL\n"); return 1;
    }
    if (n_ok != 6) { printf("  n_ok %d FAIL\n", n_ok); ok = 0; }
    if (pass1 < 0.599f || pass1 > 0.601f) { printf("  pass@1 %.3f FAIL\n", pass1); ok = 0; }
    if (fmt < 0.499f || fmt > 0.501f) { printf("  fmt %.3f FAIL\n", fmt); ok = 0; }
    /* pass@3 with 6/10: 1 - C(4,3)/C(10,3) = 1 - 4/120 = 0.9667 */
    if (passk < 0.965f || passk > 0.968f) { printf("  pass@3 %.4f FAIL\n", passk); ok = 0; }
    printf("  eval: ok %d/10 pass@1 %.3f pass@3 %.4f fmt %.3f  %s\n",
           n_ok, pass1, passk, fmt, ok ? "PASS" : "FAIL");
    printf("%s\n", ok ? "ALL EVAL TESTS PASSED" : "EVAL FAILURES");
    return ok ? 0 : 1;
}
