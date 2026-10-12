#ifndef WUBU_RESIDUAL_SCALING_H
#define WUBU_RESIDUAL_SCALING_H
/* wubu_residual_scaling.h — residual branch scaling (BRAIN #236).
 *
 * In deep residual networks, scaling each residual branch by 1/sqrt(N) (N =
 * depth) keeps the residual-stream variance stable across layers instead of
 * growing ~N. The forward (wubu_residual_scale_factor + wubu_residual_add) is
 * covered by test_residual_scaling with measured assertions.
 */
#include <stddef.h>

/* The residual branch scale factor for depth `n_layers`: 1/sqrt(n). Returns
 * 0.0 on invalid (n<=0). */
double wubu_residual_scale_factor(int n_layers);

/* x[i] = residual[i] + scale * branch[i], for n elements. Returns 0 ok. */
int wubu_residual_add(float *residual, const float *branch, int n, double scale);

#endif
