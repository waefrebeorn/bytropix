/*
 * wubu_budget_search.h -- resource-bounded mutation search (Track C4).
 *
 * Bounds a mutation search by the AD04 resource budget: accumulates
 * per-candidate cpu/ram/io cost and aborts the moment it would overrun,
 * so a search never runs away (closes the DGM "infinite sandbox" cost gap).
 */
#ifndef WUBU_BUDGET_SEARCH_H
#define WUBU_BUDGET_SEARCH_H

#include "wubu_agentic_os.h"   /* wubu_resbound_t (AD04) */

/* one candidate mutation's cost */
typedef struct {
    long cpu_ms;
    long ram_mb;
    long io_kb;
} wubu_search_cost_t;

/* Run a budgeted search over candidates[] (costs in order). Returns the
 * number evaluated before the budget stopped it (<= n). *overrun gets the
 * AD04 bitmask of the dimension that tripped (0 if the whole set fit). */
int wubu_budget_search_run(const wubu_resbound_t *budget,
                           const wubu_search_cost_t *candidates, int n,
                           int *overrun);

#endif /* WUBU_BUDGET_SEARCH_H */
