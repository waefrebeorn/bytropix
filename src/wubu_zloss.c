/* wubu_zloss.c — z-loss regularizer (BRAIN #237).
 * C11, self-contained. Numerically-stable logsumexp (max-shift); z-loss =
 * z_coef * z^2. Bounds the softmax normalizer.
 */
#include "wubu_zloss.h"
#include <math.h>

double wubu_logsumexp(const float *logits, int n) {
    if (!logits || n <= 0) return -INFINITY;
    float m = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > m) m = logits[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += exp((double)logits[i] - (double)m);
    return (double)m + log(sum);
}

double wubu_zloss(const float *logits, int n, double z_coef) {
    double z = wubu_logsumexp(logits, n);
    if (isinf(z) && z < 0) return 0.0;   /* empty/invalid -> no penalty */
    return z_coef * z * z;
}
