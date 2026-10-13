/* test_trace.c -- AGI-EDR agent tracing, OTel span per action (AGIEDR #300).
 * Proves: begin/attr/end lifecycle, parent-child linkage, unique span ids,
 * status recorded, drain oldest-first, ring wraparound, validity checks. */
#include "wubu_trace.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_tracer_t t;
    CHECK(wubu_trace_init(&t) == 0, "init ok");
    CHECK(wubu_trace_count(&t) == 0, "empty count 0");

    /* root span */
    uint64_t root = wubu_trace_begin(&t, 100, 0, 7, "run_task", 1000);
    CHECK(root > 0, "root span id > 0");
    CHECK(wubu_trace_count(&t) == 1, "count 1 after begin");

    /* child span (parent = root), same trace */
    uint64_t child = wubu_trace_begin(&t, 100, root, 7, "call_tool", 1100);
    CHECK(child > 0 && child != root, "child distinct id");
    const wubu_span_t *cs = wubu_trace_find(&t, child);
    CHECK(cs != NULL, "child findable");
    CHECK(cs->parent_id == root, "child parent == root");
    CHECK(cs->trace_id == 100, "child same trace_id");
    CHECK(strcmp(cs->name, "call_tool") == 0, "child name");
    CHECK(cs->agent_id == 7, "child agent_id");

    /* attributes */
    CHECK(wubu_trace_attr(&t, child, "tool", "web_search") == 0, "attr tool ok");
    CHECK(wubu_trace_attr(&t, child, "query", "agi edr") == 0, "attr query ok");
    cs = wubu_trace_find(&t, child);
    CHECK(cs->n_attrs == 2, "2 attrs recorded");
    CHECK(strcmp(cs->attrs[0].key, "tool") == 0 && strcmp(cs->attrs[0].val, "web_search") == 0, "attr0 key/val");

    /* end with status */
    CHECK(wubu_trace_end(&t, child, WUBU_SPAN_OK, 1500) == 0, "end child ok");
    cs = wubu_trace_find(&t, child);
    CHECK(cs->end_ns == 1500, "end_ns recorded");
    CHECK(cs->end_ns >= cs->start_ns, "end >= start");
    CHECK(cs->status == WUBU_SPAN_OK, "status OK");

    /* double-end rejected */
    CHECK(wubu_trace_end(&t, child, WUBU_SPAN_OK, 1600) == -1, "double end rejected");
    /* end unknown span rejected */
    CHECK(wubu_trace_end(&t, 99999, WUBU_SPAN_ERROR, 1600) == -1, "unknown end rejected");
    /* attr on unknown rejected */
    CHECK(wubu_trace_attr(&t, 99999, "k", "v") == -1, "unknown attr rejected");

    /* error status path */
    uint64_t err = wubu_trace_begin(&t, 100, root, 8, "risky_action", 1200);
    wubu_trace_end(&t, err, WUBU_SPAN_ERROR, 1300);
    const wubu_span_t *es = wubu_trace_find(&t, err);
    CHECK(es->status == WUBU_SPAN_ERROR, "error status recorded");

    /* monotonic clamp: end before start clamps to start */
    uint64_t bad = wubu_trace_begin(&t, 200, 0, 1, "t", 5000);
    wubu_trace_end(&t, bad, WUBU_SPAN_OK, 100);   /* now < start */
    const wubu_span_t *bs = wubu_trace_find(&t, bad);
    CHECK(bs->end_ns >= bs->start_ns, "monotonic clamp end>=start");

    /* validity */
    CHECK(wubu_span_valid(cs) == 1, "valid child span");
    CHECK(wubu_span_valid(NULL) == 0, "NULL span invalid");

    /* drain oldest-first */
    wubu_span_t out[16];
    int n = wubu_trace_drain(&t, out, 16);
    CHECK(n == wubu_trace_count(&t), "drain count matches");
    CHECK(n >= 4, "at least 4 spans drained");
    /* first drained is the oldest (root) */
    CHECK(out[0].span_id == root, "drain oldest-first (root first)");

    /* ring wraparound: emit far more than MAX */
    wubu_tracer_t big;
    wubu_trace_init(&big);
    for (int i = 0; i < WUBU_TRACE_MAX_SPANS + 50; i++) {
        char nm[16]; snprintf(nm, sizeof nm, "a%d", i);
        wubu_trace_begin(&big, 1, 0, 1, nm, (uint64_t)i);
    }
    CHECK(wubu_trace_count(&big) == WUBU_TRACE_MAX_SPANS, "ring caps at MAX");
    /* newest still findable */
    const wubu_span_t *newest = wubu_trace_find(&big, big.next_span_id - 1);
    CHECK(newest != NULL && strcmp(newest->name, "a305") == 0, "newest span retained");

    printf("\n=== test_trace: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
