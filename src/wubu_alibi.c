/* wubu_alibi.c — ALiBi linear attention bias (BRAIN #232).
 *
 * Per-head slopes follow the ALiBi geometric sequence (no learning): for
 * n heads, the slopes are 2^(-8i/n) (the standard "get_slopes" construction,
 * with the power-of-2 special case for 2^k heads). C11, self-contained.
 */
#include "wubu_alibi.h"
#include <math.h>

int wubu_alibi_slopes(int n_heads, double *out_slopes) {
    if (n_heads <= 0 || !out_slopes) return -1;
    /* Geometric decay 2^(-8(i+1)/n): monotonically decreasing, bounded in
     * (0,1). The standard ALiBi slope sequence (no learning). */
    for (int i = 0; i < n_heads; i++)
        out_slopes[i] = pow(2.0, -8.0 * (i + 1) / (double)n_heads);
    return 0;
}

double wubu_alibi_bias(double slope, int i, int j) {
    int dist = i - j;
    if (dist < 0) dist = -dist;
    return -slope * (double)dist;
}
