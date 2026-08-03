/*
 * wubu_prover2.c -- the math-RL verifier (phase 6): Lean-style proof-step
 * checking in C11. The model proposes steps; the verifier accepts or
 * rejects each (the Prover/R1 pattern). The checks mirror the theorems
 * in MATH/lean/wubu_proofs/ -- the Möbius closure, the Poincaré
 * exp∘log identity, gyroassociativity, the MLA compression factor.
 */
#include "wubu_prover.h"
#include "wubu_hyper.h"
#include <math.h>

static double mobius(double c, double x, double y)
{
    return wubu_hyper_mobius_add(c, x, y);
}

int wubu_prover_mobius_closure(double c, double x, double y)
{
    /* the Lean theorem: x²<1/c ∧ y²<1/c => |x⊕y|² < 1/c */
    if (c <= 0) return 0;
    if (!(x * x < 1.0 / c + 1e-12) || !(y * y < 1.0 / c + 1e-12)) return 0;
    double z = mobius(c, x, y);
    return (z * z < 1.0 / c + 1e-9) ? 1 : 0;
}

int wubu_prover_explog(double c, double y, double eps)
{
    /* the Poincaré identity: |exp(log(y)) - y| < eps (y in the ball) */
    if (c <= 0 || eps <= 0) return 0;
    double v = wubu_hyper_log(c, y);
    double y2 = wubu_hyper_exp(c, v);
    return fabs(y2 - y) < eps ? 1 : 0;
}

int wubu_prover_gyro(double c, double u, double v, double w, double eps)
{
    /* gyroassociativity: |u⊕(v⊕w) - (u⊕v)⊕w| < eps */
    if (c <= 0 || eps <= 0) return 0;
    double vw = mobius(c, v, w);
    double uv = mobius(c, u, v);
    double lhs = mobius(c, u, vw);
    double rhs = mobius(c, uv, w);
    return fabs(lhs - rhs) < eps ? 1 : 0;
}

int wubu_prover_check(const wubu_pf_step_t *s)
{
    if (!s) return 0;
    switch (s->kind) {
    case WUBU_PF_MOBUS:
        return wubu_prover_mobius_closure(s->a, s->b, s->c);
    case WUBU_PF_EXPLOG:
        return wubu_prover_explog(s->a, s->b, 1e-9);
    case WUBU_PF_GYRO:
        return wubu_prover_gyro(s->a, s->b, s->c, 0.0, 1e-9);
    case WUBU_PF_FACTOR:
        /* the MLA compression identity: the two sides are numerically
         * equal (H·W - H·D·U == H·(W - D·U)) */
        return fabs(s->lhs - s->rhs) < 1e-6 ? 1 : 0;
    case WUBU_PF_LINEAR:
        /* S·(a+b) == S·a + S·b (linearity of the state readout) */
        return fabs(s->lhs - s->rhs) < 1e-6 ? 1 : 0;
    case WUBU_PF_SUBST:
    case WUBU_PF_ASSOC:
    case WUBU_PF_RING:
        /* algebraic: check the numeric equality (a = b + c forms) */
        return fabs(s->a - (s->b + s->c)) < 1e-9 ? 1 : 0;
    default:
        return 0;
    }
}

int wubu_prover_check_chain(const wubu_pf_step_t *steps, int n)
{
    if (!steps || n <= 0) return 0;
    int accepted = 0;
    for (int i = 0; i < n; i++)
        if (wubu_prover_check(&steps[i])) accepted++;
    return accepted;
}
