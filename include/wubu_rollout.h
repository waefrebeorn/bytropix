/* wubu_rollout.h -- the Balanced Adaptive Rollout allocation (the
 * Orchard sparse-reward RL recipe): the rollout budget is allocated by
 * the task difficulty -- the harder tasks (the lower historical success)
 * get MORE rollouts. Deterministic + testable. */
#ifndef WUBU_ROLLOUT_H
#define WUBU_ROLLOUT_H

/* Allocate the total rollout budget across n tasks.
 * succ [n]: the historical success rates (0..1 per task; -1 = unknown).
 * budget: the total rollout count to allocate.
 * out [n]: the per-task rollout counts (sums to budget).
 * gamma: the difficulty exponent (1.0 = proportional to the failure
 *   rate; higher = more aggressive toward the hard tasks).
 * Returns 1 on success. */
int wubu_rollout_alloc(const float *succ, int n, int budget,
                       float gamma, int *out);

#endif
