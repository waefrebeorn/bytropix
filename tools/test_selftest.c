/* test_selftest.c -- KERNEL #16 kernel self-test at /n/kernel/selftest.
 * Proves: register + run tests, run_one by name, run_all summary,
 * all_passed, unknown test -> -1, fail-closed on NULL. */
#include "wubu_selftest.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

/* sample kernel self-tests */
static int st_mem_ok(void *ctx) { (void)ctx; return 0; }        /* pass */
static int st_sched_ok(void *ctx) { (void)ctx; return 0; }      /* pass */
static int st_irq_fail(void *ctx) { (void)ctx; return 1; }      /* fail */
static int st_ctx_dependent(void *ctx) { return ctx ? 0 : 1; }  /* pass if ctx set */

int main(void) {
    wubu_selftest_t st;
    CHECK(wubu_st_init(&st) == 0, "init ok");
    CHECK(wubu_st_register(&st, "mem", st_mem_ok) == 0, "register mem");
    CHECK(wubu_st_register(&st, "sched", st_sched_ok) == 0, "register sched");
    CHECK(wubu_st_register(&st, "irq", st_irq_fail) == 0, "register irq (fails)");
    CHECK(wubu_st_register(&st, "ctxdep", st_ctx_dependent) == 0, "register ctxdep");

    /* run_one by name */
    CHECK(wubu_st_run_one(&st, "mem", NULL) == 1, "run mem -> pass");
    CHECK(wubu_st_run_one(&st, "irq", NULL) == 0, "run irq -> fail");
    CHECK(wubu_st_run_one(&st, "ctxdep", NULL) == 0, "ctxdep with NULL ctx -> fail");
    CHECK(wubu_st_run_one(&st, "ctxdep", &st) == 1, "ctxdep with ctx -> pass");
    CHECK(wubu_st_run_one(&st, "bogus", NULL) == -1, "unknown test -> -1");

    /* run_all: 3 pass (mem, sched, ctxdep-with-ctx), 1 fail (irq) */
    int dummy = 1;
    int failed;
    int passed = wubu_st_run_all(&st, &dummy, &failed);
    CHECK(passed == 3, "run_all: 3 passed");
    CHECK(failed == 1, "run_all: 1 failed (irq)");
    CHECK(st.last_total == 4, "last_total 4");
    CHECK(wubu_st_all_passed(&st) == 0, "not all passed (irq failed)");

    /* a suite where everything passes */
    wubu_selftest_t ok;
    wubu_st_init(&ok);
    wubu_st_register(&ok, "a", st_mem_ok);
    wubu_st_register(&ok, "b", st_sched_ok);
    int f2;
    CHECK(wubu_st_run_all(&ok, NULL, &f2) == 2, "all-pass suite: 2 passed");
    CHECK(wubu_st_all_passed(&ok) == 1, "all_passed = 1");

    /* fail-closed */
    CHECK(wubu_st_register(&st, "bad", NULL) == -1, "NULL fn -> -1");
    CHECK(wubu_st_register(NULL, "x", st_mem_ok) == -1, "NULL registry -> -1");
    CHECK(wubu_st_run_one(NULL, "mem", NULL) == -1, "NULL run_one -> -1");
    CHECK(wubu_st_run_all(NULL, NULL, NULL) == -1, "NULL run_all -> -1");
    CHECK(wubu_st_all_passed(NULL) == 0, "NULL all_passed -> 0");

    printf("\n=== test_selftest: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
