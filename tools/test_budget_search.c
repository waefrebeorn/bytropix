/* tools/test_budget_search.c
 *
 * Resource-bounded mutation search (Track C4). Verifies the search never
 * runs away (closes the DGM "infinite sandbox" cost gap):
 *   1. a set that fits within budget evaluates all candidates
 *   2. a candidate that would overrun cpu aborts the search before paying
 *   3. the overrun bitmask names the dimension that tripped
 *   4. ram / io overruns are each caught
 */
#include "wubu_budget_search.h"
#include <stdio.h>

static int checks = 0, fails = 0;
#define CHECK(cond, msg) do { checks++; \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } } while (0)

int main(void)
{
    printf("=== test_budget_search (bounded mutation search) ===\n");
    wubu_resbound_t budget = { .cpu_ms_max = 100, .ram_mb_max = 512, .io_kb_max = 1000 };
    int overrun = 0;

    /* 1. a set that fits -> all evaluated */
    wubu_search_cost_t cheap[3] = {
        {10, 100, 100}, {20, 100, 100}, {30, 100, 100},
    };
    int done = wubu_budget_search_run(&budget, cheap, 3, &overrun);
    CHECK(done == 3, "set within budget -> all 3 evaluated");
    CHECK(overrun == 0, "no overrun when the set fits");

    /* 2. third candidate would push cpu to 10+20+90=120 > 100 -> abort at 2 */
    wubu_search_cost_t cpu_hog[3] = {
        {10, 100, 100}, {20, 100, 100}, {90, 100, 100},
    };
    done = wubu_budget_search_run(&budget, cpu_hog, 3, &overrun);
    CHECK(done == 2, "cpu overrun aborts before the 3rd candidate");
    CHECK((overrun & 1) != 0, "overrun bitmask flags cpu (bit 0)");

    /* 3. ram overrun caught */
    wubu_search_cost_t ram_hog[2] = {
        {10, 400, 100}, {10, 200, 100},   /* 400+200=600 > 512 */
    };
    done = wubu_budget_search_run(&budget, ram_hog, 2, &overrun);
    CHECK(done == 1, "ram overrun aborts before the 2nd candidate");
    CHECK((overrun & 2) != 0, "overrun bitmask flags ram (bit 1)");

    /* 4. io overrun caught */
    wubu_search_cost_t io_hog[2] = {
        {10, 100, 900}, {10, 100, 200},   /* 900+200=1100 > 1000 */
    };
    done = wubu_budget_search_run(&budget, io_hog, 2, &overrun);
    CHECK(done == 1, "io overrun aborts before the 2nd candidate");
    CHECK((overrun & 4) != 0, "overrun bitmask flags io (bit 2)");

    printf("=== test_budget_search: %d checks, %d failures ===\n", checks, fails);
    if (fails) return 1;
    printf("=== ALL BUDGET_SEARCH TESTS PASSED (the search never runs away) ===\n");
    return 0;
}
