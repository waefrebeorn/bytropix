#ifndef WUBU_BUDGET_H
#define WUBU_BUDGET_H
#include <stdint.h>

/* wubu_budget.h -- AGENT #161 resource budget bounds mutation search.
 *
 * A resource budget that BOUNDS the mutation search: caps the number of
 * mutations proposed and evaluations run, so the self-improvement loop cannot
 * consume unbounded compute. Each propose/evaluate charges the budget; when
 * exhausted, further attempts are refused (fail-closed). This is the compute
 * guard on the search. Deterministic, real work (no stubs). */

typedef struct {
    uint64_t max_mutations;   /* cap on proposed mutations */
    uint64_t max_evals;       /* cap on evaluations */
    uint64_t n_mutations;     /* mutations proposed so far */
    uint64_t n_evals;         /* evaluations run so far */
    uint64_t eval_cost;       /* cost per evaluation (default 1) */
} wubu_budget_t;

/* Initialise with mutation + eval caps. Returns 0/-1. */
int wubu_budget_init(wubu_budget_t *b, uint64_t max_mutations, uint64_t max_evals);

/* May a new mutation be proposed? (under the mutation cap). 1/0. */
int wubu_budget_may_propose(const wubu_budget_t *b);

/* Charge a mutation proposal. Returns 0 if allowed (and charges), -1 if the
 * budget is exhausted (fail-closed). */
int wubu_budget_propose(wubu_budget_t *b);

/* May an evaluation be run? (under the eval cap). 1/0. */
int wubu_budget_may_eval(const wubu_budget_t *b);

/* Charge an evaluation. Returns 0 if allowed (and charges), -1 if exhausted. */
int wubu_budget_eval(wubu_budget_t *b);

/* Is the budget fully exhausted? 1/0. */
int wubu_budget_exhausted(const wubu_budget_t *b);

/* Remaining mutation budget. */
uint64_t wubu_budget_remaining_mutations(const wubu_budget_t *b);

/* Remaining eval budget. */
uint64_t wubu_budget_remaining_evals(const wubu_budget_t *b);

#endif
