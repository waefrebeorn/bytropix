/*
 * test_db_cross.c -- O14/O18/O19 verification.
 */
#include "wubu_db_cross.h"
#include <stdio.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_db_cross (O14/O18/O19) ===\n");

    /* O14 cost-based planner: decode(0.1), prefill(5.0), evict(2.0)
     * -> order should be decode(0), evict(2), prefill(1) by ascending cost. */
    wubu_op_t ops[3] = { {1, 0.1}, {0, 5.0}, {2, 2.0} };
    int order[3];
    wubu_plan_decode(ops, 3, order);
    CHECK(order[0] == 0 && order[1] == 2 && order[2] == 1, "lowest-cost-first order");
    CHECK(wubu_plan_decode(NULL, 3, order) == 0, "null ops -> 0");

    /* O18 provable OOM-never: at 4096 ctx/batch1 the 8B KV fits 12GB; at 1M it
     * does NOT -> invariant correctly fails (would OOM without streaming). */
    int L=32, n_kv=8, d_h=128, b_kv=16;
    CHECK(wubu_kv_invariant_ok(L,n_kv,d_h,b_kv,1,4096,12e9) == 1, "4096/12GB -> ok");
    CHECK(wubu_kv_invariant_ok(L,n_kv,d_h,b_kv,1,1000000,12e9) == 0, "1M/12GB -> NOT ok (streaming needed)");
    CHECK(wubu_kv_invariant_ok(L,n_kv,d_h,b_kv,1,4096,0.0) == 0, "ram<=0 -> reject");

    /* O19 WAL replay: append 1 + mark applied, then append 2 more; replay the
     * remaining 2 (order preserved). */
    wubu_wal_t *w = wubu_wal_create(8);
    wubu_wal_append(w, 10, 4, 1);
    wubu_wal_mark_applied(w);           /* first op durable */
    wubu_wal_append(w, 11, 4, 2);
    wubu_wal_append(w, 12, 4, 3);
    wubu_wal_op_t out[8];
    int n = wubu_wal_replay(w, out, 8);
    CHECK(n == 2, "replay returns 2 unapplied");
    CHECK(out[0].block_id == 11 && out[1].block_id == 12, "replay order preserved");
    /* append beyond cap rejected */
    wubu_wal_t *w2 = wubu_wal_create(1);
    CHECK(wubu_wal_append(w2, 1, 4, 1) == 1, "append to cap=1 ok");
    CHECK(wubu_wal_append(w2, 2, 4, 2) == 0, "append beyond cap rejected");
    wubu_wal_destroy(w); wubu_wal_destroy(w2);

    if (failures == 0) { printf("ALL DB-CROSS TESTS PASSED\n"); return 0; }
    printf("%d DB-CROSS TEST(S) FAILED\n", failures);
    return 1;
}
