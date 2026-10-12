#ifndef WUBU_LABEL_SMOOTHING_H
#define WUBU_LABEL_SMOOTHING_H
/* wubu_label_smoothing.h — label smoothing (BRAIN #241).
 *
 * Converts a one-hot target (class `y` of `V`) into a smoothed distribution:
 * target[y] = 1 - alpha + alpha/V, others = alpha/V. Total mass = 1. Prevents
 * overconfident logits. The forward (wubu_label_smooth) is covered by
 * test_label_smoothing with measured assertions.
 */
#include <stddef.h>

/* Fill `target` (length V) with the smoothed distribution for class `y` under
 * smoothing `alpha` in [0,1]. Returns 0 ok, -1 bad args. */
int wubu_label_smooth(float *target, int V, int y, double alpha);

/* The cross-entropy loss of the smoothed target against log-probs `logp`
 * (length V): -sum_k target[k] * logp[k]. Returns the loss (>=0), or -1. */
double wubu_label_smooth_loss(const float *target, const float *logp, int V);

#endif
