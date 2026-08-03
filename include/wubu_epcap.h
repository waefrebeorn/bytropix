/* wubu_epcap.h -- the episode-length cap (the small-model token-budget
 * discipline from the agentic-corpus wave): a trajectory that exceeds
 * the budget is truncated at the LAST completed action boundary and
 * flagged -- the truncation keeps the budget, the flag keeps the
 * partial-credit policy. */
#ifndef WUBU_EPCAP_H
#define WUBU_EPCAP_H

/* Find the truncation point for a trajectory of n steps where each step
 * costs cost[i] tokens (0 = the step is free, e.g. a user turn).
 * budget: the total token budget.
 * out (out): the number of steps kept (the longest prefix whose cost
 *   sums to <= budget).
 * Returns 1 if the FULL trajectory fits (no truncation), 0 if it was
 *   truncated. */
int wubu_epcap(const int *cost, int n, int budget, int *out);

#endif
