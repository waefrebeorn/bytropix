#ifndef WUBU_ALIBI_H
#define WUBU_ALIBI_H
/* wubu_alibi.h — ALiBi attention bias (BRAIN #232).
 *
 * ALiBi (Press et al.) adds a linear distance penalty to attention scores
 * instead of positional embeddings: score(i,j) += -m * |i - j|, where m is a
 * per-head slope. Closer tokens attend more; distant tokens are penalized
 * linearly. The forward (wubu_alibi_bias) is covered by test_alibi with
 * measured assertions.
 */
#include <stddef.h>

/* Compute the per-head slopes for `n_heads` heads (geometric sequence).
 * Writes n_heads doubles into out_slopes. Returns 0 on ok, -1 on bad args. */
int wubu_alibi_slopes(int n_heads, double *out_slopes);

/* Compute the ALiBi bias for one head at query i, key j: bias = -m*|i-j|.
 * Returns the bias (negative or zero). */
double wubu_alibi_bias(double slope, int i, int j);

#endif
