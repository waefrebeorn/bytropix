/*
 * wubu_budget_search.c -- resource-bounded mutation search (Track C4).
 *
 * The Darwin Godel Machine's other documented gap (darwin-agents prior
 * art): "DGM didn't address the cost of failed mutations (its sandbox was
 * infinite); we have to." An unbounded search will happily burn cpu/ram/io
 * on mutations that never pay off. This module bounds it: it runs candidate
 * mutations in order, accumulating their cost, and ABORTS the search the
 * moment the cumulative cost overruns the AD04 resource budget
 * (wubu_resbound_check). A search that can't finish within budget stops
 * early and reports what it got -- it never runs away.
 *
 * Pure + dependency-light: the caller supplies each candidate's cost via a
 * callback (or the caller drives it step-by-step); this module owns the
 * budget accounting and the abort decision. Unit-testable with no real
 * workload.
 *
 * C11, minimal includes.
 */

#include "wubu_budget_search.h"
#include "wubu_agentic_os.h"   /* AD04: wubu_resbound_check */

/* Run a budgeted search. candidates[] holds per-candidate costs in order;
 * n is the count. The search evaluates candidates until the cumulative
 * cost would overrun the budget, then stops. Returns the number of
 * candidates evaluated before the budget stopped it (<= n). Fills *overrun
 * with the AD04 bitmask of the dimension that tripped (0 if the whole set
 * fit). */
int wubu_budget_search_run(const wubu_resbound_t *budget,
                           const wubu_search_cost_t *candidates, int n,
                           int *overrun) {
    if (overrun) *overrun = 0;
    if (!budget || !candidates || n <= 0) return 0;
    long cpu = 0, ram = 0, io = 0;
    for (int i = 0; i < n; i++) {
        long ncpu = cpu + candidates[i].cpu_ms;
        long nram = ram + candidates[i].ram_mb;
        long nio  = io  + candidates[i].io_kb;
        int over = wubu_resbound_check(budget, ncpu, nram, nio);
        if (over) {
            /* this candidate would overrun -- stop before paying for it */
            if (overrun) *overrun = over;
            return i;   /* evaluated i candidates, this one aborted */
        }
        cpu = ncpu; ram = nram; io = nio;
    }
    return n;   /* the whole set fit within budget */
}
