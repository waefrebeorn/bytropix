/* test_oomquar.c -- KERNEL #14 tests: memory quarantine on OOM. */
#include "wubu_oomquar.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_oomquar_t o;

    CHECK(wubu_oom_init(&o) == 0);
    CHECK(wubu_oom_add_scope(&o, "svcA", 100) == 0);
    CHECK(wubu_oom_add_scope(&o, "svcB", 200) == 0);
    CHECK(wubu_oom_add_scope(&o, "svcA", 100) == -1);   /* dup */
    CHECK(wubu_oom_find_scope(&o, "svcA") == 0);
    CHECK(wubu_oom_find_scope(&o, "nope") == -1);

    /* populate svcA: three tasks totalling 250 pages (limit 100) */
    CHECK(wubu_oom_add_task(&o, "a1", "svcA", 100) >= 0);
    CHECK(wubu_oom_add_task(&o, "a2", "svcA", 100) >= 0);
    CHECK(wubu_oom_add_task(&o, "a3", "svcA", 50) >= 0);
    /* svcB under limit */
    CHECK(wubu_oom_add_task(&o, "b1", "svcB", 100) >= 0);

    CHECK(o.scopes[0].used_pages == 250);
    CHECK(wubu_oom_over_limit(&o, "svcA") == 1);
    CHECK(wubu_oom_over_limit(&o, "svcB") == 0);

    /* handle OOM in svcA: quarantine largest tasks until <= 100.
     * largest is a1 or a2 (100 each); freeing both (200) leaves 50 <= 100. */
    int64_t rec = wubu_oom_handle(&o, "svcA");
    CHECK(rec == 200);                      /* freed a1 + a2 (100+100) */
    CHECK(o.scopes[0].used_pages == 50);    /* only a3 remains */
    CHECK(wubu_oom_over_limit(&o, "svcA") == 0);
    CHECK(o.scopes[0].oom_count == 1);
    CHECK(o.kernel_panic == 0);             /* scoped, no panic */
    CHECK(wubu_oom_kernel_panicked(&o) == 0);
    /* the two largest were quarantined, a3 survives */
    CHECK(o.tasks[0].quarantined == 1);
    CHECK(o.tasks[1].quarantined == 1);
    CHECK(o.tasks[2].quarantined == 0);

    /* svcB untouched by svcA's OOM (scoped) */
    CHECK(o.scopes[1].used_pages == 100);
    CHECK(o.tasks[3].quarantined == 0);

    /* handle OOM on a scope already under limit: no-op, returns 0 */
    CHECK(wubu_oom_handle(&o, "svcB") == 0);
    CHECK(o.scopes[1].oom_count == 0);

    /* escalation: a scope whose single task exceeds the limit and cannot be
     * brought under (freeing it still... actually freeing brings to 0 <= limit).
     * Construct a true panic: limit 0 with a task -> freeing the only task
     * leaves 0 <= 0, so no panic. Instead: a scope with limit 0 and we make
     * handle find nothing reclaimable is impossible with a task. So panic needs
     * used>limit with NO tasks -- simulate by a scope over limit with zero
     * reclaimable tasks (all already quarantined). */
    CHECK(wubu_oom_add_scope(&o, "svcC", 10) == 0);
    CHECK(wubu_oom_add_task(&o, "c1", "svcC", 5) >= 0);
    /* manually push svcC over limit with no live task by quarantining c1 then
     * inflating: simplest real panic path is a scope whose tasks all fit but
     * limit is 0 and there are tasks -> freeing them reaches 0. Test the
     * no-reclaimable path directly: add scope with limit 0, no tasks, but mark
     * over via a task that is already quarantined is not possible. So verify
     * the panic branch via a scope that is over limit with all tasks
     * quarantined: */
    {
        wubu_oomquar_t p;
        CHECK(wubu_oom_init(&p) == 0);
        CHECK(wubu_oom_add_scope(&p, "x", 5) == 0);
        int t = wubu_oom_add_task(&p, "t1", "x", 100);
        CHECK(t >= 0);
        p.tasks[t].pinned = 1;       /* kernel-resident, unreclaimable */
        int64_t r = wubu_oom_handle(&p, "x");
        CHECK(r == 0);               /* nothing reclaimable */
        CHECK(p.scopes[0].panic == 1);
        CHECK(wubu_oom_kernel_panicked(&p) == 1);
    }

    /* bad args */
    CHECK(wubu_oom_handle(&o, "nope") == -1);
    CHECK(wubu_oom_handle(NULL, "svcA") == -1);
    CHECK(wubu_oom_add_task(&o, "z", "nope", 1) == -1);
    CHECK(wubu_oom_over_limit(&o, "nope") == -1);

    /* total quarantined accumulates */
    CHECK(o.total_quarantined == 200);

    if (fails == 0) printf("ALL OOMQUAR TESTS PASSED\n");
    else printf("%d OOMQUAR TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
