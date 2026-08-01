/*
 * wubu_credit.c -- Turn-level credit assignment (AH12). C11.
 *
 * Convergence (TRACE: verifier-anchored turn-level TD credit, no critic 7-hop):
 *   - AH12: given a frozen reference model's answer-predictability before/after
 *          a tool call, assign credit: positive if predictability rose, negative
 *          if it fell, ~0 if unchanged. We approximate the reference-model probe
 *          with a cheap surrogate: a "progress score" 0..1 per turn (caller
 *          supplies; in production this is the reference-model delta). The TD
 *          credit = reward + gamma*next_value - value (standard one-step TD).
 *
 * Pure C11, deterministic, testable.
 */
#include "wubu_credit.h"
#include <stdlib.h>

/* AH12: one-step TD credit at a tool-call boundary.
 *   progress[t]  = answer-predictability after turn t (0..1)
 *   reward       = terminal outcome (1 success / 0 fail), 0 for intermediate
 *   gamma        = discount (e.g. 0.9)
 * Returns TD credit: reward + gamma*progress[t] - progress[t-1]. */
double wubu_turn_credit(double prev_progress, double cur_progress,
                        double reward, double gamma) {
    double value = cur_progress;             /* value of current state */
    return reward + gamma * value - prev_progress;
}

/* AH12: classify credit sign (for logging/attribution). */
int wubu_credit_sign(double credit, double eps) {
    if (credit >  eps) return 1;   /* positive: moved toward answer */
    if (credit < -eps) return -1;  /* negative: derailed */
    return 0;                      /* near-zero: irrelevant */
}

/* AH12: cumulate credits over a trajectory (verifier-anchored final outcome
 * still dominates; dense signal shapes learning). */
double wubu_credit_sum(const double *credits, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += credits[i];
    return s;
}
