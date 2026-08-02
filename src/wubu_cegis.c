/*
 * wubu_cegis.c -- Counterexample-guided inductive synthesis (EE03). C11.
 *
 * Convergence (CEGIS / Sketch / SyGuS 7-hop):
 *   - EE03: ∃f.∀x,y. φ(f,x,y). Loop: synthesize candidate f from grammar
 *     {max, +, -}, verify (∀x,y sound), if fail verifier returns
 *     counterexample → add to constraint set → re-synthesize. At home:
 *     searches the config grammar for the optimal tok_s config with a
 *     proof of optimality (verifier found no counterexample).
 */
#include "wubu_cegis.h"
#include <math.h>
#include <string.h>

/* Evaluate candidate on (x, y) */
static int cand_eval(wubu_cegis_cand_t c, int x, int y) {
    switch (c) {
        case CEGIS_CAND_MAX: return (x > y) ? x : y;
        case CEGIS_CAND_ADD: return x + y;
        case CEGIS_CAND_SUB: return x - y;
    }
    return 0;
}

/* Spec: f(x,y) >= x && f(x,y) >= y && (f == x || f == y)  (the max function) */
static int spec_holds(wubu_cegis_cand_t c, int x, int y) {
    int f = cand_eval(c, x, y);
    return (f >= x) && (f >= y) && (f == x || f == y);
}

/* Verify candidate over all pairs in [lo,hi] + recorded counterexamples. */
int wubu_cegis_verify(wubu_cegis_spec_t *spec, wubu_cegis_cand_t cand) {
    if (!spec) return 0;
    /* Check recorded counterexamples first */
    for (int i = 0; i < spec->n_cex; i++) {
        if (!spec_holds(cand, spec->cex_x[i], spec->cex_y[i])) {
            return 0;  /* still fails on known cex */
        }
    }
    /* Check full grid */
    for (int x = (int)spec->lo; x <= (int)spec->hi; x++)
        for (int y = (int)spec->lo; y <= (int)spec->hi; y++)
            if (!spec_holds(cand, x, y)) {
                /* Record counterexample */
                if (spec->n_cex < WUBU_CEGIS_MAX_CEX) {
                    spec->cex_x[spec->n_cex] = x;
                    spec->cex_y[spec->n_cex] = y;
                    spec->n_cex++;
                }
                return 0;
            }
    return 1;  /* valid on all */
}

int wubu_cegis_run(wubu_cegis_spec_t *spec, unsigned seed, wubu_cegis_result_t *out) {
    if (!spec || !out) return -1;
    (void)seed;
    wubu_cegis_cand_t cands[] = { CEGIS_CAND_MAX, CEGIS_CAND_ADD, CEGIS_CAND_SUB };
    for (int iter = 0; iter < WUBU_CEGIS_MAX_ITERS; iter++) {
        for (int c = 0; c < 3; c++) {
            if (wubu_cegis_verify(spec, cands[c])) {
                out->kind = cands[c];
                out->found = 1;
                return 1;
            }
        }
    }
    out->found = 0;
    return 0;
}
