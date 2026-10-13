/* test_otelauto.c -- AGIEDR #308 OTel auto-instrumentation of agent.
 * Proves: enter/exit open/close spans, nesting (parent/child), duration,
 * closed count, status, fail-closed on NULL/unbalanced. */
#include "wubu_otelauto.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_otelauto_t o;
    CHECK(wubu_oa_init(&o) == 0, "init ok");
    CHECK(wubu_oa_n_closed(&o) == 0, "0 closed spans");

    /* enter 'agent.run' */
    int h1 = wubu_oa_enter(&o, "agent.run");
    CHECK(h1 == 0, "enter agent.run -> span 0");
    CHECK(o.spans[0].parent == -1, "root span has no parent");
    wubu_oa_tick(&o, 5);

    /* nested enter 'tool.call' (child of agent.run) */
    int h2 = wubu_oa_enter(&o, "tool.call");
    CHECK(h2 == 1, "enter tool.call -> span 1");
    CHECK(o.spans[1].parent == 0, "tool.call is child of agent.run");
    wubu_oa_tick(&o, 3);

    /* exit tool.call: duration 3 */
    CHECK(wubu_oa_exit(&o, 0) == 0, "exit tool.call ok");
    CHECK(wubu_oa_span_duration(&o, 1) == 3, "tool.call duration = 3");
    CHECK(wubu_oa_n_closed(&o) == 1, "1 closed span");

    /* exit agent.run: duration 8 (5+3) */
    CHECK(wubu_oa_exit(&o, 0) == 0, "exit agent.run ok");
    CHECK(wubu_oa_span_duration(&o, 0) == 8, "agent.run duration = 8");
    CHECK(wubu_oa_n_closed(&o) == 2, "2 closed spans");

    /* status recorded */
    int h3 = wubu_oa_enter(&o, "agent.run");
    wubu_oa_exit(&o, 7);   /* error status 7 */
    CHECK(o.spans[h3].status == 7, "error status recorded");

    /* open span has duration 0 */
    int h4 = wubu_oa_enter(&o, "pending");
    CHECK(wubu_oa_span_duration(&o, h4) == 0, "open span duration 0");

    /* unbalanced exit -> -1 (pop empty stack) */
    wubu_oa_exit(&o, 0);   /* close h4 */
    CHECK(wubu_oa_exit(&o, 0) == -1, "unbalanced exit -> -1");

    /* fail-closed on NULL */
    CHECK(wubu_oa_init(NULL) == -1, "NULL init -> -1");
    CHECK(wubu_oa_enter(NULL, "x") == -1, "NULL enter -> -1");
    CHECK(wubu_oa_exit(NULL, 0) == -1, "NULL exit -> -1");
    CHECK(wubu_oa_tick(NULL, 1) == -1, "NULL tick -> -1");
    CHECK(wubu_oa_n_closed(NULL) == 0, "NULL n_closed -> 0");
    CHECK(wubu_oa_span_duration(NULL, 0) == 0, "NULL duration -> 0");

    printf("\n=== test_otelauto: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
