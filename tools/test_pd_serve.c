/*
 * test_pd_serve.c -- AB01-AB06 + AC01-AC03 verification.
 */
#include "wubu_pd_serve.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_pd_serve (AB01-AB06/AC01-AC03) ===\n");

    /* AB01 split: 2 prefill, 4 decode -> configured. */
    CHECK(wubu_pd_split(2, 4) == 1, "pd split configured");
    CHECK(wubu_pd_split(0, 4) == 0, "no prefill -> not split");

    /* AB02 handoff ready when prefill_done >= prompt_len. */
    CHECK(wubu_kv_handoff_ready(10, 10) == 1, "handoff ready at prompt_len");
    CHECK(wubu_kv_handoff_ready(9, 10) == 0, "not ready before done");

    /* AB03 pull: decode_qlen 3 < high_water 5 -> accept. */
    CHECK(wubu_pull_route(3, 5) == 1, "decode accepts below high-water");
    CHECK(wubu_pull_route(5, 5) == 0, "decode full -> reject");

    /* AB04 hetero map: prefill=0 (compute), decode=1 (bw). */
    int pt, dt; wubu_hetero_map(&pt, &dt);
    CHECK(pt == 0 && dt == 1, "prefill compute tier, decode bw tier");

    /* AB05 xfer fits: 1e9 bytes / 1e10 B/s = 0.1s <= 0.2s budget -> fits. */
    CHECK(wubu_kv_xfer_fits(1e9, 1e10, 0.2) == 1, "xfer fits TTFT budget");
    CHECK(wubu_kv_xfer_fits(1e9, 1e9, 0.2) == 0, "xfer too slow -> no fit");

    /* AB06 prefix reuse on hash match. */
    CHECK(wubu_prefix_reuse(12345u, 12345u) == 1, "prefix hash hit -> reuse");
    CHECK(wubu_prefix_reuse(12345u, 99999u) == 0, "mismatch -> no reuse");

    /* AC01 MoD: gate 0.8 >= thr 0.5 -> execute. */
    CHECK(wubu_mod_execute(0.8f, 0.5f) == 1, "MoD gate high -> execute");
    CHECK(wubu_mod_execute(0.2f, 0.5f) == 0, "MoD gate low -> skip");

    /* AC02 capacity: depth 8, cap 4 -> keep 4. */
    CHECK(wubu_mod_capacity(8, 4) == 4, "MoD cap depth to 4");
    CHECK(wubu_mod_capacity(2, 4) == 2, "MoD under cap -> keep 2");

    /* AC03 early-exit: conf 0.95 >= thr 0.9 -> exit. */
    CHECK(wubu_early_exit(0.95f, 0.9f) == 1, "high conf -> early exit");
    CHECK(wubu_early_exit(0.5f, 0.9f) == 0, "low conf -> continue");

    if (failures == 0) { printf("ALL PD-SERVE TESTS PASSED\n"); return 0; }
    printf("%d PD-SERVE TEST(S) FAILED\n", failures);
    return 1;
}
