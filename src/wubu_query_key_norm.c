/* wubu_query_key_norm.c — QK RMSNorm (BRAIN #235).
 * C11, self-contained. RMSNorm (no mean subtraction): x / sqrt(mean(x^2)+eps)
 * * gamma. Bounds the RMS of Q/K to ~gamma, capping attention logits.
 */
#include "wubu_query_key_norm.h"
#include <math.h>

int wubu_qk_rmsnorm(float *x, int n, float gamma, float eps) {
    if (n <= 0 || !x || eps <= 0.0f) return -1;
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    double mean = ss / (double)n;
    double inv = 1.0 / sqrt(mean + (double)eps);
    for (int i = 0; i < n; i++) x[i] = (float)((double)x[i] * inv * (double)gamma);
    return 0;
}
