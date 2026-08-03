/*
 * wubu_hyper.c -- the WuBu hyperbolic layer (OUR math, Lean-verified).
 *
 * All formulas ported from MATH/lean/wubu_proofs/:
 *   mobius_add_1d c x y = ((1 + 2cx·y + c·y²)·x + (1 - c·x²)·y)
 *                         / (1 + 2cx·y + c²·x²·y²)
 *   exp_0^c(v)  = tanh(√c·‖v‖) · v / (√c·‖v‖)
 *   log_0^c(y)  = atanh(√c·‖y‖) · y / (√c·‖y‖)
 * The vector Möbius addition (gyrovector space) is the standard
 * generalization; the 1D case is proven in Lean.
 */
#include "wubu_hyper.h"
#include <math.h>
#include <string.h>

double wubu_hyper_mobius_add(double c, double x, double y)
{
    /* the Lean-verified formula (denominator provably nonzero for
     * ball points) */
    double num = (1.0 + 2.0 * c * x * y + c * y * y) * x
               + (1.0 - c * x * x) * y;
    double den = 1.0 + 2.0 * c * x * y + c * c * x * x * y * y;
    return num / den;
}

double wubu_hyper_exp(double c, double v)
{
    double sc = sqrt(c);
    double nv = fabs(v);
    if (nv < 1e-12) return 0.0;
    double t = tanh(sc * nv) / (sc * nv);
    return t * v;
}

double wubu_hyper_log(double c, double y)
{
    double sc = sqrt(c);
    double ny = fabs(y);
    if (ny < 1e-12) return 0.0;
    double t = atanh(sc * ny) / (sc * ny);
    return t * y;
}

void wubu_hyper_add_vec(double c, const double *x, const double *y,
                        double *out, int n)
{
    /* the gyrovector Möbius addition in n dims:
     * x ⊕ y = ( (1 + 2c<x,y> + c‖y‖²)·x + (1 - c‖x‖²)·y )
     *        / ( 1 + 2c<x,y> + c²‖x‖²·‖y‖² ) */
    double xy = 0, nx2 = 0, ny2 = 0;
    for (int i = 0; i < n; i++) { xy += x[i] * y[i]; nx2 += x[i] * x[i]; ny2 += y[i] * y[i]; }
    double den = 1.0 + 2.0 * c * xy + c * c * nx2 * ny2;
    double a = (1.0 + 2.0 * c * xy + c * ny2) / den;
    double b = (1.0 - c * nx2) / den;
    for (int i = 0; i < n; i++) out[i] = a * x[i] + b * y[i];
}

void wubu_hyper_exp_vec(double c, const double *v, double *out, int n)
{
    double sc = sqrt(c);
    double nv = 0;
    for (int i = 0; i < n; i++) nv += v[i] * v[i];
    nv = sqrt(nv);
    if (nv < 1e-12) { memset(out, 0, n * sizeof(double)); return; }
    double t = tanh(sc * nv) / (sc * nv);
    for (int i = 0; i < n; i++) out[i] = t * v[i];
}

void wubu_hyper_log_vec(double c, const double *y, double *out, int n)
{
    double sc = sqrt(c);
    double ny = 0;
    for (int i = 0; i < n; i++) ny += y[i] * y[i];
    ny = sqrt(ny);
    if (ny < 1e-12) { memset(out, 0, n * sizeof(double)); return; }
    double t = atanh(sc * ny) / (sc * ny);
    for (int i = 0; i < n; i++) out[i] = t * y[i];
}

double wubu_hyper_dist(double c, const double *x, const double *y, int n)
{
    /* d(x,y) = (2/√c) · atanh(√c · ‖(-x) ⊕_c y‖) */
    if (n <= 0 || n > 64) return -1;
    double negx[64], sum[64];
    for (int i = 0; i < n; i++) negx[i] = -x[i];
    wubu_hyper_add_vec(c, negx, y, sum, n);
    double nsum = 0;
    for (int i = 0; i < n; i++) nsum += sum[i] * sum[i];
    nsum = sqrt(nsum);
    return (2.0 / sqrt(c)) * atanh(sqrt(c) * nsum);
}

int wubu_hyper_lift(const wubu_hyper_t *h, const double *x, double *y)
{
    if (!h || !x || !y || h->n <= 0 || h->n > 64) return -1;
    wubu_hyper_exp_vec(h->c, x, y, h->n);
    return 0;
}

int wubu_hyper_project(const wubu_hyper_t *h, const double *y, double *x)
{
    if (!h || !y || !x || h->n <= 0 || h->n > 64) return -1;
    wubu_hyper_log_vec(h->c, y, x, h->n);
    return 0;
}

int wubu_hyper_gyro_rotate(const wubu_hyper_t *h,
                           const double *q, const double *k, double *out)
{
    /* align the query to the key on the ball: q' = (-k) ⊕_c q --
     * the gyration, the relative query in the key's frame. The 1D
     * gyration is the identity (HyperbolicGyration.lean); the vector
     * case uses the Möbius addition. */
    if (!h || !q || !k || !out || h->n <= 0 || h->n > 64) return -1;
    double negk[64];
    for (int i = 0; i < h->n; i++) negk[i] = -k[i];
    wubu_hyper_add_vec(h->c, negk, q, out, h->n);
    return 0;
}
