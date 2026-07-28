#ifndef WUBU_MHC_H
#define WUBU_MHC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_mhc wubu_mhc_t;

/* Manifold-Constrained Hyper-Connections: widened residual streams (exp) over
 * base hidden dim, with sigmoid-nonnegative pre/post projections and a
 * manifold-constrained mixing matrix (identity-preserving at init). */
wubu_mhc_t *wubu_mhc_create(int dim, int exp);
void wubu_mhc_free(wubu_mhc_t *m);
void wubu_mhc_apply_nonneg(float *w, int n);
int  wubu_mhc_identity_ok(const wubu_mhc_t *m);
void wubu_mhc_set_identity(wubu_mhc_t *m);
void wubu_mhc_forward(const wubu_mhc_t *m, const float *x, float *r_out, float *y_out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MHC_H */
