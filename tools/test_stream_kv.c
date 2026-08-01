/*
 * test_stream_kv.c -- StreamingLLM attention-sink remap verification (L01).
 *
 * Verifies: identity when disabled, live-set = {0..sink-1} U {len-window..},
 * capacity bound, edge cases (cap==0, window>=len, sink>len, pos out of range)
 * and a round-trip that every physical slot is hit exactly once.
 */
#include "wubu_stream_kv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

static void test_identity_disabled(void) {
    wubu_stream_kv_t *s = wubu_stream_kv_create(0, 0);
    CHECK(wubu_stream_kv_enabled(s) == 0, "disabled when sink=win=0");
    CHECK(wubu_stream_kv_remap(s, 1000, 500) == 500, "identity remap");
    CHECK(wubu_stream_kv_capacity(s, 1000) == 1000, "identity capacity");
    CHECK(wubu_stream_kv_live_count(s, 1000) == 1000, "identity live");
    wubu_stream_kv_destroy(s);
}

static void test_basic_sink_window(void) {
    /* sink=4, window=8, len=20 -> live = {0,1,2,3} + {12..19} = 12 tokens */
    int sink = 4, window = 8, len = 20;
    wubu_stream_kv_t *s = wubu_stream_kv_create(sink, window);
    CHECK(wubu_stream_kv_enabled(s) == 1, "enabled");
    CHECK(wubu_stream_kv_capacity(s, len) == sink + window, "capacity = sink+window");
    CHECK(wubu_stream_kv_live_count(s, len) == 12, "live count = 12");

    /* sink tokens map to themselves */
    for (int p = 0; p < sink; p++)
        CHECK(wubu_stream_kv_remap(s, len, p) == p, "sink maps to self");
    /* middle evicted */
    CHECK(wubu_stream_kv_remap(s, len, 5) == -1, "middle evicted (5)");
    CHECK(wubu_stream_kv_remap(s, len, 11) == -1, "middle evicted (11)");
    /* window maps contiguously */
    CHECK(wubu_stream_kv_remap(s, len, 12) == sink + 0, "window start -> sink");
    CHECK(wubu_stream_kv_remap(s, len, 19) == sink + window - 1, "window end -> cap-1");
    /* out of range */
    CHECK(wubu_stream_kv_remap(s, len, -1) == -1, "neg pos");
    CHECK(wubu_stream_kv_remap(s, len, len) == -1, "pos==len");
    wubu_stream_kv_destroy(s);
}

static void test_window_covers_all(void) {
    /* window >= len => nothing evicted (full cache) */
    int len = 50;
    wubu_stream_kv_t *s = wubu_stream_kv_create(4, 100);
    CHECK(wubu_stream_kv_capacity(s, len) == len, "window>=len => cap=len");
    CHECK(wubu_stream_kv_live_count(s, len) == len, "window>=len => all live");
    for (int p = 0; p < len; p++)
        CHECK(wubu_stream_kv_remap(s, len, p) == p, "all identity when window covers");
    wubu_stream_kv_destroy(s);
}

static void test_sink_gt_len(void) {
    /* sink > len => everything is sink, nothing evicted */
    int len = 10;
    wubu_stream_kv_t *s = wubu_stream_kv_create(64, 0);
    CHECK(wubu_stream_kv_live_count(s, len) == len, "sink>len => all live");
    for (int p = 0; p < len; p++)
        CHECK(wubu_stream_kv_remap(s, len, p) == p, "sink>len identity");
    wubu_stream_kv_destroy(s);
}

static void test_roundtrip_surjective(void) {
    /* Every physical slot [0,cap) is the image of exactly one live token. */
    int sink = 4, window = 8, len = 100;
    wubu_stream_kv_t *s = wubu_stream_kv_create(sink, window);
    int cap = wubu_stream_kv_capacity(s, len);
    int *hit = (int *)calloc(cap, sizeof(int));
    int live = 0;
    for (int p = 0; p < len; p++) {
        int slot = wubu_stream_kv_remap(s, len, p);
        if (slot >= 0) {
            CHECK(slot >= 0 && slot < cap, "slot in range");
            hit[slot]++;
            live++;
        }
    }
    for (int i = 0; i < cap; i++)
        CHECK(hit[i] == 1, "each physical slot hit exactly once (surjective+injective)");
    CHECK(live == wubu_stream_kv_live_count(s, len), "live count matches remap");
    free(hit);
    wubu_stream_kv_destroy(s);
}

int main(void) {
    printf("=== test_stream_kv (L01 StreamingLLM attention-sink) ===\n");
    test_identity_disabled();
    test_basic_sink_window();
    test_window_covers_all();
    test_sink_gt_len();
    test_roundtrip_surjective();
    if (failures == 0) {
        printf("ALL STREAM-KV TESTS PASSED\n");
        return 0;
    }
    printf("%d STREAM-KV TEST(S) FAILED\n", failures);
    return 1;
}
