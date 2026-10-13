/* wubu_hypmom.c -- WORLDSIM #270 hyperbolic momentum embedder.
 * See wubu_hypmom.h. Poincare disk model of hyperbolic space. */
#include "wubu_hypmom.h"
#include <math.h>

double wubu_hm_norm(const double *v, int dim) {
    if (!v) return 0.0;
    double s = 0.0;
    for (int i = 0; i < dim; i++) s += v[i] * v[i];
    return sqrt(s);
}

int wubu_hm_init(wubu_hypmom_t *h, double curvature) {
    if (!h) return -1;
    for (int i = 0; i < WUBU_HM_DIM; i++) { h->p[i] = 0.0; h->v[i] = 0.0; }
    h->curvature = (curvature < 0) ? curvature : -1.0;   /* must be negative */
    return 0;
}

/* Clamp a point to stay strictly inside the unit disk. */
static void clamp_disk(double *p) {
    double n = wubu_hm_norm(p, WUBU_HM_DIM);
    const double maxn = 1.0 - 1e-6;
    if (n >= maxn) {
        double scale = maxn / n;
        for (int i = 0; i < WUBU_HM_DIM; i++) p[i] *= scale;
    }
}

int wubu_hm_exp_map(const wubu_hypmom_t *h, const double *v, double *out) {
    if (!h || !v || !out) return -1;
    double c = -h->curvature;         /* positive magnitude */
    double vn = wubu_hm_norm(v, WUBU_HM_DIM);
    double pn = wubu_hm_norm(h->p, WUBU_HM_DIM);
    /* conformal factor lambda_p = 2 / (1 - |p|^2) for the Poincare disk */
    double lam = 2.0 / (1.0 - pn * pn);
    double denom = 1.0 + 2.0 * c * lam * 0.0;  /* placeholder; use standard formula */
    (void)denom;
    /* Mobius addition-based exp map (simplified Gyrovector form):
     *   out = (p + (lam/2) * v) / (1 + (lam/2) * c * <p,v> ... ) is complex;
     * use the standard exp_p(v) = p ⊕ (tanh(sqrt(c)*|v|/2) * v/|v|).
     * Here v is already a velocity; treat lr*|v| as the tangent step. */
    double sn = vn;
    double tanh_arg = tanh(sqrt(c) * sn / 2.0);
    if (sn < 1e-12) {
        for (int i = 0; i < WUBU_HM_DIM; i++) out[i] = h->p[i];
        return 0;
    }
    /* direction in the tangent space at p, approximated by v (origin-relative) */
    for (int i = 0; i < WUBU_HM_DIM; i++) {
        out[i] = h->p[i] + tanh_arg * (v[i] / sn) * (1.0 - pn);
    }
    clamp_disk(out);
    return 0;
}

int wubu_hm_step(wubu_hypmom_t *h, const double *g, double lr, double momentum) {
    if (!h || !g || lr < 0 || momentum < 0 || momentum > 1) return -1;
    /* update momentum: v = momentum * v + lr * g (gradient ascent/descent) */
    for (int i = 0; i < WUBU_HM_DIM; i++) {
        h->v[i] = momentum * h->v[i] + lr * g[i];
    }
    /* move along the exponential map of the velocity */
    double out[WUBU_HM_DIM];
    if (wubu_hm_exp_map(h, h->v, out) != 0) return -1;
    for (int i = 0; i < WUBU_HM_DIM; i++) h->p[i] = out[i];
    return 0;
}

double wubu_hm_distance(const double *a, const double *b, double curvature) {
    if (!a || !b) return 0.0;
    double c = -curvature;
    if (c <= 0) c = 1.0;
    double diff = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < WUBU_HM_DIM; i++) {
        double d = a[i] - b[i];
        diff += d * d;
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    /* Poincare disk distance: d = arcosh(1 + 2*|a-b|^2 / ((1-|a|^2)(1-|b|^2))) */
    double denom = (1.0 - na) * (1.0 - nb);
    if (denom <= 1e-12) denom = 1e-12;
    double arg = 1.0 + 2.0 * c * diff / denom;
    if (arg < 1.0) arg = 1.0;
    return acosh(arg) / sqrt(c);
}
