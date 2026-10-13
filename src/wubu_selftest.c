/* wubu_selftest.c -- KERNEL #16 kernel self-test at /n/kernel/selftest.
 * See wubu_selftest.h. */
#include "wubu_selftest.h"
#include <string.h>

int wubu_st_init(wubu_selftest_t *st) {
    if (!st) return -1;
    memset(st->tests, 0, sizeof st->tests);
    st->n = 0;
    st->last_total = 0;
    st->last_passed = 0;
    st->last_failed = 0;
    return 0;
}

int wubu_st_register(wubu_selftest_t *st, const char *name, wubu_st_fn fn) {
    if (!st || !name || !fn || st->n >= WUBU_ST_MAX) return -1;
    if (strlen(name) >= WUBU_ST_NAMELEN) return -1;
    strncpy(st->tests[st->n].name, name, WUBU_ST_NAMELEN - 1);
    st->tests[st->n].name[WUBU_ST_NAMELEN - 1] = '\0';
    st->tests[st->n].fn = fn;
    st->n++;
    return 0;
}

int wubu_st_run_one(wubu_selftest_t *st, const char *name, void *ctx) {
    if (!st || !name) return -1;
    for (int i = 0; i < st->n; i++) {
        if (strcmp(st->tests[i].name, name) == 0) {
            return st->tests[i].fn(ctx) == 0 ? 1 : 0;
        }
    }
    return -1;   /* unknown test */
}

int wubu_st_run_all(wubu_selftest_t *st, void *ctx, int *out_failed) {
    if (!st) return -1;
    int passed = 0, failed = 0;
    for (int i = 0; i < st->n; i++) {
        if (st->tests[i].fn(ctx) == 0) passed++;
        else failed++;
    }
    st->last_total = st->n;
    st->last_passed = passed;
    st->last_failed = failed;
    if (out_failed) *out_failed = failed;
    return passed;
}

int wubu_st_all_passed(const wubu_selftest_t *st) {
    if (!st) return 0;
    return st->last_failed == 0 ? 1 : 0;
}
