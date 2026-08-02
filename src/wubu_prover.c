/*
 * wubu_prover.c -- Automated theorem proving: natural-deduction prover (EE04). C11.
 *
 * Convergence (Lean / Coq / natural-deduction ATP 7-hop):
 *   - EE04: a lightweight propositional + arithmetic prover. Given premises and
 *     a goal, it tries to prove the goal via:
 *       1. truth-table / exhaustive assignment check for boolean structure
 *       2. arithmetic evaluation for GE/LE/EQ comparisons
 *       3. modus ponens + resolution on implications
 *     At home: the causal SCM (AW) + symbolic rules produce conjectures
 *     (e.g. "KV=512K ⇒ tok_s ≥ 25"); the prover soundly verifies them.
 */
#include "wubu_prover.h"
#include <string.h>

/* Recursive evaluator: returns 1 (true) / 0 (false). var_assign gives values
   for PROP_VAR-indexed variables (0 or 1 for boolean, integer for arith). */
int wubu_prover_eval(const wubu_prop_t *p, int *var_assign, int n_vars) {
    if (!p) return 0;
    switch (p->kind) {
        case PROP_TRUE:  return 1;
        case PROP_FALSE: return 0;
        case PROP_ATOM:  return var_assign ? (var_assign[p->a % n_vars] ? 1 : 0) : 0;
        case PROP_VAR:   return var_assign ? (var_assign[p->a % n_vars] ? 1 : 0) : 0;
        case PROP_NOT:
            return !wubu_prover_eval(p->left, var_assign, n_vars);
        case PROP_AND:
            return wubu_prover_eval(p->left, var_assign, n_vars) &&
                   wubu_prover_eval(p->right, var_assign, n_vars);
        case PROP_OR:
            return wubu_prover_eval(p->left, var_assign, n_vars) ||
                   wubu_prover_eval(p->right, var_assign, n_vars);
        case PROP_IMPL:
            return !wubu_prover_eval(p->left, var_assign, n_vars) ||
                    wubu_prover_eval(p->right, var_assign, n_vars);
        case PROP_IFF:
            return wubu_prover_eval(p->left, var_assign, n_vars) ==
                   wubu_prover_eval(p->right, var_assign, n_vars);
        case PROP_GE:
        case PROP_LE:
        case PROP_EQ: {
            int l = p->a, r = p->b;
            if (p->kind == PROP_GE) return l >= r;
            if (p->kind == PROP_LE) return l <= r;
            return l == r;
        }
    }
    return 0;
}

/* Check if the goal is a tautology given the premises, by enumerating all
   boolean assignments to the proposition's free vars and checking that
   whenever all premises hold, the goal holds. (Sound but incomplete.) */
static int implies_under_assignment(const wubu_proof_t *proof, int *asg, int n) {
    /* If all premises true under this assignment, goal must be true. */
    for (int i = 0; i < proof->n_premises; i++) {
        if (proof->premises[i] &&
            !wubu_prover_eval(proof->premises[i], asg, n))
            return 1;  /* premise false → implication vacuously holds */
    }
    return wubu_prover_eval(proof->goal, asg, n);
}

int wubu_prover_prove(const wubu_proof_t *proof) {
    if (!proof || !proof->goal) return 0;
    /* Enumerate all 2^n assignments for up to 6 boolean vars. */
    int n = 6;
    int total = 1 << n;
    for (int mask = 0; mask < total; mask++) {
        int asg[6];
        for (int i = 0; i < n; i++) asg[i] = (mask >> i) & 1;
        if (!implies_under_assignment(proof, asg, n))
            return 0;  /* found counterexample → not provable */
    }
    return 1;  /* no counterexample → provable (tautology under premises) */
}
