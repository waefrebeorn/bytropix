/*
 * wubu_hyper.h -- the WuBu hyperbolic layer (OUR math, Lean-verified).
 *
 * Phase 1 of the WuBu Model blueprint: the nested-hyperbolic inductive
 * bias. Every formula here is ported from our FORMAL proofs in
 * MATH/lean/wubu_proofs/ -- the Möbius addition closure, the
 * exp_0^c / log_0^c maps, and the gyration, all proven in Lean:
 *
 *   MobiusAdd.lean          mobius_add_1d preserves the Poincaré ball
 *   PoincareBall.lean       exp_0^c(log_0^c(y)) = y
 *   HyperbolicGyration.lean gyroassociativity + ball preservation
 *
 * The model lifts the hidden stream into a Poincaré ball with learnable
 * curvature c_i, computes attention with gyro-rotated Q/K, and projects
 * back -- hierarchy computed in the right geometry. The math is PROVEN,
 * not assumed.
 */
#ifndef WUBU_HYPER_H
#define WUBU_HYPER_H

#include <stdint.h>

/* ---- the proven 1D operations (scalar curvature c > 0) ---- */

/* Möbius addition: x ⊕_c y. Lean: preserves the ball (x²<1/c, y²<1/c
 * => (x⊕y)² < 1/c) and the denominator is provably nonzero. */
double wubu_hyper_mobius_add(double c, double x, double y);

/* exp_0^c(v): lift a tangent vector v into the ball. */
double wubu_hyper_exp(double c, double v);

/* log_0^c(y): project a ball point y back to the tangent space.
 * Lean: exp(log(y)) == y (the PoincareBall identity). */
double wubu_hyper_log(double c, double y);

/* ---- the vectorized operations (n-dim, used by the model) ---- */

/* Möbius addition in n dims (gyrovector space). out[i] = x ⊕_c y.
 * The scalar gyration is the identity in 1D (HyperbolicGyration.lean);
 * the vector version uses the standard gyrovector formula. */
void wubu_hyper_add_vec(double c, const double *x, const double *y,
                        double *out, int n);

/* exp_0^c of a tangent vector v: tanh(√c·‖v‖) · v / (√c·‖v‖). */
void wubu_hyper_exp_vec(double c, const double *v, double *out, int n);

/* log_0^c of a ball point y: atanh(√c·‖y‖) · y / (√c·‖y‖). */
void wubu_hyper_log_vec(double c, const double *y, double *out, int n);

/* the Poincaré distance between two ball points (the hyperbolic metric
 * the model optimizes). */
double wubu_hyper_dist(double c, const double *x, const double *y, int n);

/* ---- the model-facing layer ---- */

/* The hyperbolic lift block: x (Euclidean, [n]) -> lift into the ball
 * with curvature c, gyro-rotate the query/key for attention, project
 * back. This is the layer the WuBu blocks call when hyperbolic mode
 * is on. Returns 0 on success. */
typedef struct {
    double c;        /* learnable curvature (> 0) */
    int    n;        /* dim */
    /* the level descriptor ld_i (WuBu Nesting) */
    double ld[64];
} wubu_hyper_t;

/* lift: y = exp_0^c(x)  (ball point) */
int wubu_hyper_lift(const wubu_hyper_t *h, const double *x, double *y);
/* project: x = log_0^c(y)  (back to tangent) */
int wubu_hyper_project(const wubu_hyper_t *h, const double *y, double *x);
/* gyro-rotate q by k (the attention query alignment on the ball) */
int wubu_hyper_gyro_rotate(const wubu_hyper_t *h,
                           const double *q, const double *k, double *out);

#endif
