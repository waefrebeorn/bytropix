/* wubu_budget.c -- AGENT #161 resource budget bounds mutation search.
 * See wubu_budget.h. */
#include "wubu_budget.h"

int wubu_budget_init(wubu_budget_t *b, uint64_t max_mutations, uint64_t max_evals) {
    if (!b) return -1;
    b->max_mutations = max_mutations;
    b->max_evals = max_evals;
    b->n_mutations = 0;
    b->n_evals = 0;
    b->eval_cost = 1;
    return 0;
}

int wubu_budget_may_propose(const wubu_budget_t *b) {
    if (!b) return 0;
    return b->n_mutations < b->max_mutations ? 1 : 0;
}

int wubu_budget_propose(wubu_budget_t *b) {
    if (!b || b->n_mutations >= b->max_mutations) return -1;
    b->n_mutations++;
    return 0;
}

int wubu_budget_may_eval(const wubu_budget_t *b) {
    if (!b) return 0;
    return b->n_evals < b->max_evals ? 1 : 0;
}

int wubu_budget_eval(wubu_budget_t *b) {
    if (!b || b->n_evals >= b->max_evals) return -1;
    b->n_evals++;
    return 0;
}

int wubu_budget_exhausted(const wubu_budget_t *b) {
    if (!b) return 1;
    return (b->n_mutations >= b->max_mutations && b->n_evals >= b->max_evals) ? 1 : 0;
}

uint64_t wubu_budget_remaining_mutations(const wubu_budget_t *b) {
    if (!b || b->n_mutations >= b->max_mutations) return 0;
    return b->max_mutations - b->n_mutations;
}

uint64_t wubu_budget_remaining_evals(const wubu_budget_t *b) {
    if (!b || b->n_evals >= b->max_evals) return 0;
    return b->max_evals - b->n_evals;
}
