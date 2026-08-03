/* wubu_eval.h -- the agentic eval harness (the AC-H eval side): the
 * trajectory-level evaluation that COMPOSES the modules -- the DB-state
 * verification (wubu_db_verify), the pass@k (wubu_passk), and the
 * format-validity rate (the wubu_fmt checks). */
#ifndef WUBU_EVAL_H
#define WUBU_EVAL_H

#include "wubu_dbstate.h"

typedef struct {
    const wubu_db_slot_t *state;   /* the trajectory's final state */
    int nslots;
    int format_ok;                 /* 1 = every assistant turn parsed
                                      (precomputed via wubu_fmt) */
} wubu_eval_traj_t;

/* Run the eval: goals [n] are the per-trajectory annotated goals; the
 * trajectory t's final state is verified against goals[t] (the DB-state
 * check). out:
 *   n_ok     -- the number of trajectories whose state met the goal
 *   pass1    -- the pass@1 (n_ok/n)
 *   passk    -- the pass@k (the unbiased estimator over the n attempts)
 *   fmt_rate -- the fraction of trajectories with format_ok
 * Returns 1 on success. */
int wubu_eval_run(const wubu_db_goal_t *goals, const wubu_eval_traj_t *trajs,
                  int n, int k, int *n_ok, float *pass1, float *passk,
                  float *fmt_rate);

#endif
