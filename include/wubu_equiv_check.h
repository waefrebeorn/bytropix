/*
 * wubu_equiv_check.h — Lightweight GEMV equivalence verifier (doc F02, CPU core).
 *
 * Alive2 formally proves LLVM rewrites preserve semantics. We cannot run Alive2
 * (external LLVM tooling), but the *spirit* — verify two implementations of the
 * same math produce equivalent results — is pure C11. This checks two GEMV
 * routines (e.g. a naive vs a SIMD/tile-rewritten one) over random inputs and
 * reports max abs / relative error against a tolerance. Ties to 009.
 */
#ifndef WUBU_EQUIV_CHECK_H
#define WUBU_EQUIV_CHECK_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Compare two [n] result vectors; return max abs diff and max rel diff (via
 * out pointers). Returns 0 if both within tol, -1 otherwise. */
int wubu_equiv_vectors(const float *a, const float *b, int n,
                        float tol_abs, float *max_abs, float *max_rel);

/* Verify a GEMV impl `fn` against a naive reference over `trials` random inputs
 * of dim `n`. fn(out, W, x, n) computes out = W (row n) dot x. Returns max
 * relative error observed; prints a summary. */
typedef void (*wubu_gemv_fn)(float *out, const float *W, const float *x, int n);
float wubu_equiv_gemv(wubu_gemv_fn fn, int n, int trials, float tol_rel);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_EQUIV_CHECK_H */
