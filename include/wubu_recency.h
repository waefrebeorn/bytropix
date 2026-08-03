/* wubu_recency.h -- the recency-weighted sampling (the CAI corpus
 * freshness cadence): the corpus position i of n gets the weight
 *   w(i) = base + (1 - base) * (i/n)^power
 * -- the freshest data is weighted highest, the base floor keeps the
 * old data alive. The corpus-mix recipe's freshness axis. */
#ifndef WUBU_RECENCY_H
#define WUBU_RECENCY_H

/* Weight of the position i in a stream of n tokens.
 * base: the floor weight (0..1, e.g. 0.2); power: the recency curve
 *   (1.0 = linear, >1 = the fresh data even more dominant).
 * Returns the weight in [base, 1]. */
float wubu_recency_weight(long i, long n, float base, float power);

#endif
