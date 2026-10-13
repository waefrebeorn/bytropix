/* test_ftrace.c -- KERNEL #30 tests: ftrace ring buffer as /n/kernel/ftrace. */
#include "wubu_ftrace.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_ftrace_t f;
    wubu_ftrace_rec_t r;

    /* init: disabled, empty */
    CHECK(wubu_ftrace_init(&f) == 0);
    CHECK(f.enabled == 0);
    CHECK(wubu_ftrace_count(&f) == 0);

    /* disabled: events are no-ops */
    CHECK(wubu_ftrace_enter(&f, "main", 1) == 0);
    CHECK(wubu_ftrace_count(&f) == 0);
    CHECK(f.total == 0);

    /* enable + a nested call sequence: main -> foo -> bar */
    CHECK(wubu_ftrace_set_enabled(&f, 1) == 0);
    CHECK(wubu_ftrace_enter(&f, "main", 10) == 0);   /* depth 0 */
    CHECK(wubu_ftrace_enter(&f, "foo", 20) == 0);    /* depth 1 */
    CHECK(wubu_ftrace_enter(&f, "bar", 30) == 0);    /* depth 2 */
    CHECK(f.cur_depth == 3);
    CHECK(wubu_ftrace_exit(&f, "bar", 40) == 0);     /* depth 2 */
    CHECK(wubu_ftrace_exit(&f, "foo", 50) == 0);     /* depth 1 */
    CHECK(wubu_ftrace_exit(&f, "main", 60) == 0);    /* depth 0 */
    CHECK(f.cur_depth == 0);
    CHECK(wubu_ftrace_count(&f) == 6);

    /* chronological read order + fields */
    CHECK(wubu_ftrace_read(&f, 0, &r) == 0);
    CHECK(r.timestamp == 10 && r.depth == 0 && r.is_exit == 0 && strcmp(r.func, "main") == 0);
    CHECK(wubu_ftrace_read(&f, 1, &r) == 0);
    CHECK(r.timestamp == 20 && r.depth == 1 && strcmp(r.func, "foo") == 0);
    CHECK(wubu_ftrace_read(&f, 3, &r) == 0);
    CHECK(r.timestamp == 40 && r.depth == 2 && r.is_exit == 1 && strcmp(r.func, "bar") == 0);
    CHECK(wubu_ftrace_read(&f, 6, &r) == -1);   /* out of range */

    /* dump contains all six, in order */
    {
        char buf[4096];
        int n = wubu_ftrace_dump(&f, buf, sizeof buf);
        CHECK(n > 0);
        CHECK(strstr(buf, "{ main") != NULL);
        CHECK(strstr(buf, "} main") != NULL);
        /* main's entry appears before foo's entry */
        char *pm = strstr(buf, "{ main");
        char *pf = strstr(buf, "{ foo");
        CHECK(pm && pf && pm < pf);
    }

    /* clear resets everything */
    CHECK(wubu_ftrace_clear(&f) == 0);
    CHECK(wubu_ftrace_count(&f) == 0);
    CHECK(f.total == 0 && f.dropped == 0 && f.cur_depth == 0);

    /* ring overflow: write CAP+10 entries, oldest 10 dropped */
    for (int i = 0; i < WUBU_FTRACE_CAP + 10; i++) {
        char nm[16]; snprintf(nm, sizeof nm, "f%d", i);
        wubu_ftrace_enter(&f, nm, (uint64_t)i);
        wubu_ftrace_exit(&f, nm, (uint64_t)i);
    }
    CHECK(wubu_ftrace_count(&f) == WUBU_FTRACE_CAP);   /* ring full */
    CHECK(f.total == (uint64_t)(WUBU_FTRACE_CAP + 10) * 2);
    CHECK(f.dropped == (uint64_t)(WUBU_FTRACE_CAP + 10) * 2 - WUBU_FTRACE_CAP); /* 276 */
    /* oldest retained is event #276 = the enter of f138 (f0..f137 fully dropped) */
    CHECK(wubu_ftrace_read(&f, 0, &r) == 0);
    CHECK(strcmp(r.func, "f138") == 0 && r.is_exit == 0);
    /* newest retained is the last event = exit of f265 */
    CHECK(wubu_ftrace_read(&f, WUBU_FTRACE_CAP - 1, &r) == 0);
    CHECK(strcmp(r.func, "f265") == 0 && r.is_exit == 1);

    /* disable mid-trace stops recording but keeps the buffer */
    CHECK(wubu_ftrace_set_enabled(&f, 0) == 0);
    int before = wubu_ftrace_count(&f);
    wubu_ftrace_enter(&f, "ignored", 9999);
    CHECK(wubu_ftrace_count(&f) == before);
    CHECK(f.total == (uint64_t)(WUBU_FTRACE_CAP + 10) * 2);  /* unchanged */

    /* NULL fail-closed */
    CHECK(wubu_ftrace_init(NULL) == -1);
    CHECK(wubu_ftrace_enter(NULL, "x", 1) == -1);
    CHECK(wubu_ftrace_count(NULL) == 0);
    CHECK(wubu_ftrace_read(&f, -1, &r) == -1);

    if (fails == 0) printf("ALL FTRACE TESTS PASSED\n");
    else printf("%d FTRACE TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
