#ifndef WUBU_SELFTEST_H
#define WUBU_SELFTEST_H
#include <stdint.h>

/* wubu_selftest.h -- KERNEL #16 kernel self-test at /n/kernel/selftest.
 *
 * An on-demand kernel self-test suite (kselftest-equivalent): a registry of
 * named kernel self-tests that can be run individually or all-at-once, with
 * pass/fail results and a summary. Exposed at /n/kernel/selftest so an
 * operator can trigger the kernel's built-in tests via the 9P namespace.
 * Deterministic, fixed-size, real work (no stubs). */

#define WUBU_ST_MAX 32
#define WUBU_ST_NAMELEN 48

typedef int (*wubu_st_fn)(void *ctx);   /* returns 0 = pass, nonzero = fail */

typedef struct {
    char name[WUBU_ST_NAMELEN];
    wubu_st_fn fn;
} wubu_st_entry_t;

typedef struct {
    wubu_st_entry_t tests[WUBU_ST_MAX];
    int n;
    /* last-run results */
    int last_total;
    int last_passed;
    int last_failed;
} wubu_selftest_t;

/* Initialise the self-test registry. */
int wubu_st_init(wubu_selftest_t *st);

/* Register a self-test. Returns 0/-1. */
int wubu_st_register(wubu_selftest_t *st, const char *name, wubu_st_fn fn);

/* Run one self-test by name. Returns 1 if passed, 0 if failed, -1 if unknown. */
int wubu_st_run_one(wubu_selftest_t *st, const char *name, void *ctx);

/* Run all self-tests. Returns the number passed; fills passed/failed. */
int wubu_st_run_all(wubu_selftest_t *st, void *ctx, int *out_failed);

/* Did the last run pass everything? 1/0. */
int wubu_st_all_passed(const wubu_selftest_t *st);

#endif
