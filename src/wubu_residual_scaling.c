/* wubu_residual_scaling.c — residual branch scaling (BRAIN #236).
 * C11, self-contained. Scale = 1/sqrt(n_layers); residual += scale*branch.
 */
#include "wubu_residual_scaling.h"
#include <math.h>

double wubu_residual_scale_factor(int n_layers) {
    if (n_layers <= 0) return 0.0;
    return 1.0 / sqrt((double)n_layers);
}

int wubu_residual_add(float *residual, const float *branch, int n, double scale) {
    if (!residual || !branch || n <= 0) return -1;
    for (int i = 0; i < n; i++)
        residual[i] = (float)((double)residual[i] + scale * (double)branch[i]);
    return 0;
}
