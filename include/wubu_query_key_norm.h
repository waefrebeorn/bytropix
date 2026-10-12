#ifndef WUBU_QUERY_KEY_NORM_H
#define WUBU_QUERY_KEY_NORM_H
/* wubu_query_key_norm.h — Query-Key normalization (BRAIN #235).
 *
 * QK-Norm applies RMSNorm to Q and K projections before the attention dot
 * product, bounding the logit magnitude and stabilizing training. The forward
 * (wubu_qk_rmsnorm) is covered by test_qk_norm with measured assertions.
 */
#include <stddef.h>

/* In-place RMSNorm of `n` values with scale `gamma` and epsilon `eps`.
 * out[i] = x[i] / sqrt(mean(x^2) + eps) * gamma. Returns 0 on ok, -1 on bad
 * args (n<=0, NULL, eps<=0). */
int wubu_qk_rmsnorm(float *x, int n, float gamma, float eps);

#endif
