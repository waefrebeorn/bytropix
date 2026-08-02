/*
 * wubu_prover.h -- Automated theorem proving: natural-deduction prover (EE04).
 */
#ifndef WUBU_PROVER_H
#define WUBU_PROVER_H

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

#endif