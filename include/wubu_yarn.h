#ifndef WUBU_YARN_H
#define WUBU_YARN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* YaRN NTK-aware RoPE extrapolation. Computes per-dim scale + ramp vectors. */
int wubu_yarn_scales(int d, double L_train, double L_target,
                     double beta, double *scale, double *ramp);
/* Apply scaling to a rotary angle for a given dim index. */
void wubu_yarn_apply(double theta, const double *scale, int half, int dim_idx,
                     double *theta_out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_YARN_H */
