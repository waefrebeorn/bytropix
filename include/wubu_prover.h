/*
 * wubu_prover.h -- automated theorem proving for the WuBu AGI.
 *
 * TWO prover APIs:
 *
 * 1. EE04 natural-deduction prover (the existing engine): propositional
 *    + arithmetic proofs via truth-table enumeration + evaluation.
 *    `wubu_prover_prove(premises, goal)` -- sound but incomplete.
 *
 * 2. The math-RL verifier (phase 6): Lean-style proof-step checking in
 *    C11. The model proposes proof STEPS; the verifier accepts or
 *    rejects each one (the Prover/R1 pattern: correct proofs are the
 *    reward). Checks the Möbius closure, the Poincaré exp∘log identity,
 *    gyroassociativity, the MLA compression factor identity -- the
 *    same theorems we proved in MATH/lean/wubu_proofs/.
 */
#ifndef WUBU_PROVER_H
#define WUBU_PROVER_H

#include <stdint.h>

#define WUBU_PROVER_MAX_PREMS 16
#define WUBU_PROVER_MAX_ATOMS 64

typedef enum {
    PROP_TRUE, PROP_FALSE,
    PROP_ATOM,            /* atomic predicate, index into atoms */
    PROP_NOT, PROP_AND, PROP_OR, PROP_IMPL, PROP_IFF,
    PROP_GE, PROP_LE, PROP_EQ,  /* arithmetic comparison on integer vars */
    PROP_VAR              /* a named variable */
} wubu_prop_kind_t;

typedef struct wubu_prop {
    wubu_prop_kind_t kind;
    int a, b;      /* child prop indices, or atom index, or var index */
    int c, d;      /* extra (e.g. arithmetic operands) */
    struct wubu_prop *left, *right;
} wubu_prop_t;

typedef struct {
    wubu_prop_t *premises[WUBU_PROVER_MAX_PREMS];
    int n_premises;
    wubu_prop_t *goal;
} wubu_proof_t;

/* Attempt to prove goal from premises using natural deduction + truth-table
   for boolean combinators and arithmetic evaluation for comparisons.
   Returns 1 if provable, 0 otherwise. */
int wubu_prover_prove(const wubu_proof_t *proof);

/* Simplify / evaluate a proposition under an assignment. */
int wubu_prover_eval(const wubu_prop_t *p, int *var_assign, int n_vars);

/* ================ the math-RL verifier (phase 6) ================ */

/* the proof-step kinds the verifier understands */
typedef enum {
    WUBU_PF_SUBST = 0,   /* replace a subexpression with an equal one */
    WUBU_PF_ASSOC,       /* reassociate a product/sum */
    WUBU_PF_FACTOR,      /* H·W - H·D·U -> H·(W - D·U) (the MLA theorem) */
    WUBU_PF_MOBUS,       /* the Möbius closure: |x⊕y|² < 1/c */
    WUBU_PF_EXPLOG,      /* the Poincaré identity: exp(log(y)) = y */
    WUBU_PF_GYRO,        /* gyroassociativity: u⊕(v⊕w) = (u⊕v)⊕w */
    WUBU_PF_LINEAR,      /* linear attention: S·(a+b) = S·a + S·b */
    WUBU_PF_RING,        /* ring simplification (the Lean `ring` tactic) */
    WUBU_PF_COUNT
} wubu_pf_kind_t;

/* one proof step */
typedef struct {
    wubu_pf_kind_t kind;
    /* the operands (for the numerical checks) */
    double a, b, c;
    /* for the mla factor check: the "compression identity" holds when
     * the two sides are numerically equal */
    double lhs, rhs;
} wubu_pf_step_t;

/* P1: verify ONE step. Returns 1 (accepted), 0 (rejected). */
int wubu_prover_check(const wubu_pf_step_t *s);

/* P2: verify a CHAIN of steps (a proof). Returns the number accepted
 * (a full proof = all accepted). The model's reward. */
int wubu_prover_check_chain(const wubu_pf_step_t *steps, int n);

/* P3: the Möbius closure (the Lean theorem in C11): given x,y in the
 * ball (|x|²,|y|² < 1/c), verify |x⊕y|² < 1/c. */
int wubu_prover_mobius_closure(double c, double x, double y);

/* P4: the Poincaré identity: |exp(log(y)) - y| < eps. */
int wubu_prover_explog(double c, double y, double eps);

/* P5: the gyroassociativity: |u⊕(v⊕w) - (u⊕v)⊕w| < eps. */
int wubu_prover_gyro(double c, double u, double v, double w, double eps);

#endif
