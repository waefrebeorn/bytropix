/*
 * wubu_mhc_mh.h -- Multi-head Hyper-Connections (the 2512.24880 form).
 *
 * The existing wubu_mhc.c is the Round-3 SIGMOID variant (one widened
 * residual stream, dim*exp, sigmoid-nonneg pre/post projections). This
 * module is the PAPER-form MULTI-HEAD variant: a GROUP of nh hidden
 * states h[0..nh-1] each of dim d, a learned nh x nh mixing matrix M
 * whose rows are softmax-constrained (the manifold constraint -- convex
 * combination), a gated write, and an EXACT identity initialization
 * (M = one-hot, alpha = 1) that makes the module behave like a plain
 * residual connection (the function-preserving oracle).
 *
 * Distinct symbol names (wubu_mhc_mh_*) -- the two variants coexist.
 * C11, self-contained, no third-party deps.
 */
#ifndef WUBU_MHC_MH_H
#define WUBU_MHC_MH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_mhc_mh wubu_mhc_mh_t;

/* nh = number of parallel hidden streams; d = dim of each stream.
 * Mixing matrix M is [nh x nh], initialized deterministically (seed). */
wubu_mhc_mh_t *wubu_mhc_mh_create(int nh, int d, uint32_t seed);
void wubu_mhc_mh_free(wubu_mhc_mh_t *m);

/* Exact-identity init: M = one-hot rows + alpha default 1.0 -- the
 * module then behaves EXACTLY like a plain residual (the oracle). */
void wubu_mhc_mh_set_identity(wubu_mhc_mh_t *m);

/* The manifold constraint: row-softmax every row of M so each row is a
 * convex combination (sums to 1, stays in [0,1]). */
void wubu_mhc_mh_constrain(wubu_mhc_mh_t *m);

/* READ: x_in = sum_k M[i,k] * h[k]  (h is an array of nh pointers, each
 * d floats). Writes out[d]. Returns 0 on success. */
int wubu_mhc_mh_read(const wubu_mhc_mh_t *m, const float *const *h,
                     int i, float *out);

/* WRITE: h_i = alpha*h_i + (1-alpha)*y  (the gated write). Returns 0. */
int wubu_mhc_mh_write(wubu_mhc_mh_t *m, float *h_i, const float *y,
                      float alpha);

/* accessors */
int wubu_mhc_mh_nh(const wubu_mhc_mh_t *m);
int wubu_mhc_mh_dim(const wubu_mhc_mh_t *m);
const float *wubu_mhc_mh_mixing_row(const wubu_mhc_mh_t *m, int i);
/* returns 1 if every row sums to 1 within 1e-4 (the manifold holds) */
int wubu_mhc_mh_manifold_ok(const wubu_mhc_mh_t *m);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MHC_MH_H */
