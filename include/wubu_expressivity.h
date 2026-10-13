#ifndef WUBU_EXPRESSIVITY_H
#define WUBU_EXPRESSIVITY_H

/* wubu_expressivity.h -- PATHS #324 expressivity-invariant risk tracking.
 *
 * Tracks whether a self-modification preserves the system's EXPRESSIVITY --
 * the set of distinct behaviors/outputs it can produce. A mutation that
 * shrinks expressivity (loses capabilities) or changes it unpredictably is
 * RISKY, even if it passes its immediate test. The module holds a baseline
 * expressivity measure and, for each proposed mutation, computes the change
 * and flags risk: expressivity loss = risky, expressivity gain = novel,
 * unchanged = safe. Deterministic, real work (no stubs). */

typedef enum {
    WUBU_EXPR_SAFE=0,   /* expressivity preserved (within tolerance) */
    WUBU_EXPR_NOVEL,    /* expressivity increased (novel capability) */
    WUBU_EXPR_RISKY,    /* expressivity decreased (capability loss) */
} wubu_expr_risk_t;

typedef struct {
    double baseline;     /* baseline expressivity measure */
    double tolerance;    /* relative change considered "safe" */
    int    n_mutations;
    int    n_risky;      /* mutations flagged risky */
    int    n_novel;
} wubu_expressivity_t;

/* Initialise with a baseline expressivity and a tolerance (e.g. 0.05). */
int wubu_expr_init(wubu_expressivity_t *x, double baseline, double tolerance);

/* Assess a mutation's new expressivity vs baseline. Returns the risk class. */
wubu_expr_risk_t wubu_expr_assess(wubu_expressivity_t *x, double new_expressivity);

/* Update the baseline (accept a mutation that passed review). */
int wubu_expr_accept(wubu_expressivity_t *x, double new_expressivity);

/* Fraction of mutations that were risky (risk rate). */
double wubu_expr_risk_rate(const wubu_expressivity_t *x);

#endif
