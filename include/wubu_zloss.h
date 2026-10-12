#ifndef WUBU_ZLOSS_H
#define WUBU_ZLOSS_H
/* wubu_zloss.h — z-loss regularizer (BRAIN #237).
 *
 * Z-loss penalizes the log-partition (logsumexp) of the logits: z = logsumexp,
 * loss += z_coef * z^2. It keeps the softmax normalizer bounded, preventing
 * logit blow-up in large-vocab models. The forward (wubu_logsumexp +
 * wubu_zloss) is covered by test_zloss with measured assertions.
 */
#include <stddef.h>

/* Numerically-stable logsumexp of `n` logits. Returns the value. */
double wubu_logsumexp(const float *logits, int n);

/* The z-loss term: z_coef * (logsumexp)^2. Returns the penalty (>=0). */
double wubu_zloss(const float *logits, int n, double z_coef);

#endif
