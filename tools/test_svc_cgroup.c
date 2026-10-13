/* test_svc_cgroup.c -- KERNEL #15 tests: cgroup v2 per-service via /n/svc. */
#include "wubu_svc_cgroup.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_svc_cgroups_t s;

    CHECK(wubu_svc_init(&s) == 0);
    CHECK(wubu_svc_add(&s, "web") == 0);
    CHECK(wubu_svc_add(&s, "db") == 0);
    CHECK(wubu_svc_add(&s, "web") == -1);          /* dup */
    CHECK(wubu_svc_find(&s, "web") == 0);
    CHECK(wubu_svc_find(&s, "nope") == -1);

    /* cpu controller */
    CHECK(wubu_svc_set_cpu(&s, "web", 50000, 100000) == 0);  /* 50% of a core */
    CHECK(wubu_svc_set_cpu(&s, "web", 100, 0) == -1);        /* period must be >0 */
    CHECK(wubu_svc_set_cpu(&s, "web", -1, 100000) == -1);    /* quota >= 0 */
    CHECK(wubu_svc_set_cpu(&s, "nope", 1, 1) == -1);

    /* account cpu: under quota not throttled, at/over throttled */
    CHECK(wubu_svc_account(&s, "web", 40000, 0, 0) == 0);
    CHECK(wubu_svc_cpu_throttled(&s, "web") == 0);
    CHECK(wubu_svc_account(&s, "web", 10000, 0, 0) == 0);    /* used = 50000 = quota */
    CHECK(wubu_svc_cpu_throttled(&s, "web") == 1);

    /* unlimited cpu (quota 0) never throttles */
    CHECK(wubu_svc_set_cpu(&s, "db", 0, 100000) == 0);
    CHECK(wubu_svc_account(&s, "db", 999999, 0, 0) == 0);
    CHECK(wubu_svc_cpu_throttled(&s, "db") == 0);

    /* memory controller */
    CHECK(wubu_svc_set_mem(&s, "web", 1000) == 0);
    CHECK(wubu_svc_account(&s, "web", 0, 500, 0) == 0);
    CHECK(wubu_svc_mem_exceeded(&s, "web") == 0);
    CHECK(wubu_svc_account(&s, "web", 0, 500, 0) == 0);      /* used = 1000 = max */
    CHECK(wubu_svc_mem_exceeded(&s, "web") == 0);            /* not > max */
    CHECK(wubu_svc_account(&s, "web", 0, 1, 0) == 0);        /* used = 1001 > max */
    CHECK(wubu_svc_mem_exceeded(&s, "web") == 1);

    /* unlimited memory (max 0) never exceeded */
    CHECK(wubu_svc_set_mem(&s, "db", 0) == 0);
    CHECK(wubu_svc_account(&s, "db", 0, 999999, 0) == 0);
    CHECK(wubu_svc_mem_exceeded(&s, "db") == 0);

    /* io controller */
    CHECK(wubu_svc_set_io(&s, "web", 2000) == 0);
    CHECK(wubu_svc_account(&s, "web", 0, 0, 1500) == 0);
    CHECK(wubu_svc_io_exceeded(&s, "web") == 0);
    CHECK(wubu_svc_account(&s, "web", 0, 0, 600) == 0);      /* used = 2100 > 2000 */
    CHECK(wubu_svc_io_exceeded(&s, "web") == 1);
    CHECK(wubu_svc_set_io(&s, "db", 0) == 0);
    CHECK(wubu_svc_account(&s, "db", 0, 0, 999999) == 0);
    CHECK(wubu_svc_io_exceeded(&s, "db") == 0);

    /* per-service independence: cache limits don't affect web */
    CHECK(wubu_svc_add(&s, "cache") == 0);
    CHECK(wubu_svc_set_mem(&s, "cache", 10) == 0);
    CHECK(wubu_svc_mem_exceeded(&s, "web") == 1);   /* web still over its own */
    CHECK(wubu_svc_mem_exceeded(&s, "cache") == 0); /* fresh, under limit */

    /* bad args */
    CHECK(wubu_svc_set_mem(&s, "nope", 100) == -1);
    CHECK(wubu_svc_set_io(&s, "nope", 100) == -1);
    CHECK(wubu_svc_account(&s, "nope", 1, 1, 1) == -1);
    CHECK(wubu_svc_cpu_throttled(&s, "nope") == -1);
    CHECK(wubu_svc_mem_exceeded(&s, "nope") == -1);
    CHECK(wubu_svc_io_exceeded(&s, "nope") == -1);

    if (fails == 0) printf("ALL SVC_CGROUP TESTS PASSED\n");
    else printf("%d SVC_CGROUP TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
