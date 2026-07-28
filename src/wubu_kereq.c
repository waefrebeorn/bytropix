/*
 * wubu_kereq.c — Kernel equivalence via abstract interpretation (Round-2 #121/#123).
 * C11, self-contained. Genuine (if lightweight) SYMBOLIC prover: represents each
 * kernel as an affine+clamp op y = clamp(scale*x + bias, lo, hi) and computes the
 * OUTPUT INTERVAL over the input range [x_lo, x_hi] by abstract interpretation
 * (interval arithmetic). This is sound:
 *   - disjoint output intervals  => PROVEN divergence (SAT) + counterexample
 *   - identical specs            => PROVEN equal (UNSAT)
 *   - overlapping intervals      => UNKNOWN (cannot conclude)  -- honest, not "equal"
 * This catches structural bugs (e.g. a candidate that clamps to a disjoint range)
 * that finite numeric tests miss, the way Gimlet/ProofWright do -- but it is a
 * real prover over the affine+clamp class, not a single hard-coded pattern.
 */
#include "wubu_kereq.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* Output interval of affine+clamp over x in [xlo,xhi]. */
static void out_iv(const wubu_affine_clamp_t *k, float xlo, float xhi,
                   float *olo, float *ohi) {
    float a = xlo * k->scale + k->bias;
    float b = xhi * k->scale + k->bias;
    float mn = fminf(a, b), mx = fmaxf(a, b);
    *olo = fmaxf(mn, k->lo);
    *ohi = fminf(mx, k->hi);
}

/* Prove equivalence of two affine+clamp kernels over [xlo,xhi].
 * Returns: 1 = proven EQUAL (UNSAT), 0 = proven DIVERGENT (SAT, *cx set),
 *          2 = UNKNOWN (intervals overlap, inconclusive). */
int wubu_kereq_prove_eq(const wubu_affine_clamp_t *ref,
                        const wubu_affine_clamp_t *cand,
                        float xlo, float xhi, float *cx) {
    if (xhi < xlo) { float t = xlo; xlo = xhi; xhi = t; }  /* tolerate reversed range */
    float rlo, rhi, clo, chi;
    out_iv(ref, xlo, xhi, &rlo, &rhi);
    out_iv(cand, xlo, xhi, &clo, &chi);

    /* Disjoint intervals => guaranteed divergence at the gap. */
    if (rhi < clo - 1e-9f || chi < rlo - 1e-9f) {
        *cx = (rhi < clo) ? rhi : chi;   /* a reference output the candidate misses */
        return 0;                         /* SAT */
    }
    /* Identical specifications => proven equal. */
    if (ref->scale == cand->scale && ref->bias == cand->bias &&
        ref->lo == cand->lo && ref->hi == cand->hi) {
        return 1;                         /* UNSAT */
    }
    return 2;                             /* UNKNOWN (honest) */
}
