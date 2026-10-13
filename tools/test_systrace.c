/* test_systrace.c -- KERNEL #19 syscall trace streams to a 9P fifo.
 * Proves: record syscalls, exit sets retval, FIFO drain order, pending count,
 * ring overflow drops oldest, fail-closed on NULL. */
#include "wubu_systrace.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_systrace_t t;
    CHECK(wubu_strace_init(&t) == 0, "init ok");

    /* record a few syscalls */
    int64_t a1[4] = { 3, 0x1000, 512, 0 };
    uint64_t s1 = wubu_strace_enter(&t, 100, 0 /*read*/, a1);
    CHECK(s1 == 1, "first syscall seq 1");
    wubu_strace_exit(&t, 512);   /* read returned 512 */
    int64_t a2[4] = { 1, 0x2000, 256, 0 };
    wubu_strace_enter(&t, 100, 1 /*write*/, a2);
    wubu_strace_exit(&t, 256);
    CHECK(wubu_strace_pending(&t) == 2, "2 pending records");

    /* drain FIFO: first record has seq 1, retval 512 */
    wubu_st_rec_t r;
    CHECK(wubu_strace_read(&t, &r) == 1, "read one record");
    CHECK(r.seq == 1 && r.nr == 0 && r.retval == 512, "first: seq1 read ret 512");
    CHECK(r.args[1] == 0x1000, "arg preserved");
    CHECK(wubu_strace_read(&t, &r) == 1, "read second");
    CHECK(r.seq == 2 && r.nr == 1 && r.retval == 256, "second: seq2 write ret 256");

    /* FIFO empty -> 0 */
    CHECK(wubu_strace_read(&t, &r) == 0, "empty FIFO -> 0");

    /* timestamps monotonic */
    wubu_strace_enter(&t, 1, 9, NULL);
    uint64_t ts_a = t.ring[(t.tail-1+WUBU_ST_CAP)%WUBU_ST_CAP].timestamp;
    wubu_strace_enter(&t, 1, 9, NULL);
    uint64_t ts_b = t.ring[(t.tail-1+WUBU_ST_CAP)%WUBU_ST_CAP].timestamp;
    CHECK(ts_b > ts_a, "timestamps monotonic increasing");

    /* ring overflow: fill beyond CAP, oldest dropped */
    for (int i = 0; i < WUBU_ST_CAP + 5; i++) wubu_strace_enter(&t, 1, 9, NULL);
    CHECK(t.dropped >= 5, "overflow dropped oldest records");
    CHECK(wubu_strace_pending(&t) == WUBU_ST_CAP, "capped at CAP pending");

    /* fail-closed */
    CHECK(wubu_strace_enter(NULL, 1, 0, NULL) == 0, "NULL enter -> 0");
    CHECK(wubu_strace_exit(NULL, 0) == -1, "NULL exit -> -1");
    CHECK(wubu_strace_read(NULL, &r) == -1, "NULL read -> -1");
    CHECK(wubu_strace_pending(NULL) == 0, "NULL pending -> 0");
    CHECK(wubu_strace_init(NULL) == -1, "NULL init -> -1");

    printf("\n=== test_systrace: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
