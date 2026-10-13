/* test_bootbudget.c -- KERNEL #22 boot time budgeted + reported.
 * Proves: phase timing, breach flagged, totals, within-budget, fail-closed. */
#include "wubu_bootbudget.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_bootbudget_t b;
    CHECK(wubu_bb_init(&b) == 0, "init ok");
    CHECK(wubu_bb_add(&b, "firmware", 100) == 0, "add firmware (budget 100)");
    CHECK(wubu_bb_add(&b, "pci", 50) == 0, "add pci (budget 50)");
    CHECK(wubu_bb_add(&b, "mem", 80) == 0, "add mem (budget 80)");
    CHECK(wubu_bb_add(&b, "drivers", 200) == 0, "add drivers (budget 200)");
    CHECK(wubu_bb_total_budget(&b) == 430, "total budget = 430");

    /* all within budget */
    CHECK(wubu_bb_record(&b, "firmware", 90) == 0, "firmware 90 <= 100 ok");
    CHECK(wubu_bb_record(&b, "pci", 40) == 0, "pci 40 <= 50 ok");
    CHECK(wubu_bb_record(&b, "mem", 75) == 0, "mem 75 <= 80 ok");
    CHECK(wubu_bb_record(&b, "drivers", 180) == 0, "drivers 180 <= 200 ok");
    CHECK(wubu_bb_total(&b) == 385, "total actual = 385");
    CHECK(wubu_bb_breaches(&b) == 0, "0 breaches");
    CHECK(wubu_bb_within(&b) == 1, "boot within total budget");

    /* a phase breaches its budget */
    CHECK(wubu_bb_record(&b, "pci", 120) == 1, "pci 120 > 50 -> BREACH");
    CHECK(wubu_bb_breaches(&b) == 1, "1 breach");

    /* two phases breach; total may still be under sum-of-budgets or not */
    CHECK(wubu_bb_record(&b, "mem", 200) == 1, "mem 200 > 80 -> BREACH");
    CHECK(wubu_bb_breaches(&b) == 2, "2 breaches");
    /* total actual now 90+120+200+180 = 590 > 430 budget */
    CHECK(wubu_bb_total(&b) == 590, "total actual = 590");
    CHECK(wubu_bb_within(&b) == 0, "boot OVER total budget");

    /* recording an unknown phase -> -1 */
    CHECK(wubu_bb_record(&b, "nope", 10) == -1, "unknown phase -> -1");

    /* fail-closed */
    CHECK(wubu_bb_add(NULL, "x", 1) == -1, "NULL add -> -1");
    CHECK(wubu_bb_record(NULL, "x", 1) == -1, "NULL record -> -1");
    CHECK(wubu_bb_total(NULL) == 0, "NULL total -> 0");
    CHECK(wubu_bb_breaches(NULL) == 0, "NULL breaches -> 0");
    CHECK(wubu_bb_within(NULL) == 0, "NULL within -> 0 (fail-closed)");
    CHECK(wubu_bb_init(NULL) == -1, "NULL init -> -1");

    printf("\n=== test_bootbudget: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
