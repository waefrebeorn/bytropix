/* wubu_eval.c -- the agentic eval harness. */
#include <stdlib.h>
#include "wubu_eval.h"
#include "wubu_passk.h"

int wubu_eval_run(const wubu_db_goal_t *goals, const wubu_eval_traj_t *trajs,
                  int n, int k, int *n_ok, float *pass1, float *passk,
                  float *fmt_rate)
{
    if (!goals || !trajs || n < 1 || k < 1) return 0;
    int okc = 0, fmts = 0;
    int *succ = (int *)malloc((size_t)n * sizeof(int));
    if (!succ) return 0;
    for (int t = 0; t < n; t++) {
        int v = wubu_db_verify(&goals[t], 1, trajs[t].state, trajs[t].nslots);
        succ[t] = (v == 1) ? 1 : 0;
        if (succ[t]) okc++;
        if (trajs[t].format_ok) fmts++;
    }
    if (n_ok) *n_ok = okc;
    if (pass1) *pass1 = (float)okc / (float)n;
    if (passk) *passk = wubu_passk(succ, n, k > n ? n : k);
    if (fmt_rate) *fmt_rate = (float)fmts / (float)n;
    free(succ);
    return 1;
}
