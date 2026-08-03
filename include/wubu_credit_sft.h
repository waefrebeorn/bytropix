/* wubu_credit_sft.h -- the credit-assignment SFT (the Orchard recipe):
 * learn from the PRODUCTIVE SEGMENTS of even UNRESOLVED trajectories.
 * A trajectory is a list of (segment, success_flag) steps; a productive
 * segment is one that advances the state toward the goal (success=1 up to
 * and including the last success -- the "winning prefix"). The credit
 * mask = 1 on the productive segments, 0 on the failure tail. */
#ifndef WUBU_CREDIT_SFT_H
#define WUBU_CREDIT_SFT_H

/* Compute the credit mask for a trajectory of n steps.
 * succ [n]: 1 = the step succeeded / advanced, 0 = it failed.
 * mask [n] (out): 1 on the productive prefix (the leading run of 1s and
 *   the single trailing 1 if present -- see below), 0 on the failure tail.
 * Returns the number of credited steps. */
int wubu_credit_mask(const int *succ, int n, int *mask);

/* The credit fraction (credited / n) -- the partial-credit ratio. */
float wubu_credit_frac(const int *mask, int n);

#endif
