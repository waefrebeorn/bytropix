#ifndef WUBU_HYPMOM_H
#define WUBU_HYPMOM_H

/* wubu_hypmom.h -- WORLDSIM #270 hyperbolic momentum embedder.
 *
 * Embeds worldsim state into a hyperbolic space (the Poincare disk), which
 * represents hierarchy/tree-like structure with exponentially more room than
 * Euclidean space. The embedder maps a state vector into the disk via the
 * exponential map, applies a momentum update in the tangent space, and
 * projects back (keeping points strictly inside the disk). Distance in the
 * disk is the hyperbolic metric. Deterministic, real work (no stubs). */

#define WUBU_HM_DIM 8

typedef struct {
    double p[WUBU_HM_DIM];    /* point in the Poincare disk (norm < 1) */
    double v[WUBU_HM_DIM];    /* momentum (tangent velocity) */
    double curvature;         /* negative curvature c (|c|>0), e.g. -1 */
} wubu_hypmom_t;

/* Initialise a point at the origin with zero momentum. */
int wubu_hm_init(wubu_hypmom_t *h, double curvature);

/* Exponential map at the current point: exp_p(v) -> a new point in the disk. */
int wubu_hm_exp_map(const wubu_hypmom_t *h, const double *v, double *out);

/* Parallel-transport-free momentum step: p <- exp_p(lr * v), then update
 * momentum toward a target gradient (g) with friction. Returns 0/-1. */
int wubu_hm_step(wubu_hypmom_t *h, const double *g, double lr, double momentum);

/* Hyperbolic distance between two points in the disk. */
double wubu_hm_distance(const double *a, const double *b, double curvature);

/* Norm of a vector. */
double wubu_hm_norm(const double *v, int dim);

#endif
