/* wubu_passk.h -- the pass@k estimator (the Orchard/Claw-Eval metric):
 * the probability that at least one of k attempts succeeds. The unbiased
 * combinatorial estimate over n attempts with s successes:
 *   pass@k = 1 - C(n-s, k) / C(n, k)     (sample without replacement) */
#ifndef WUBU_PASSK_H
#define WUBU_PASSK_H

/* succ [n]: the per-attempt success flags (0/1).
 * k: the attempts allowed (1..n).
 * Returns the unbiased pass@k in [0,1] (0 when n < k or no attempts). */
float wubu_passk(const int *succ, int n, int k);

#endif
