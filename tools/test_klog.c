/* test_klog.c -- KERNEL #18 structured kernel log at /n/kernel/klog.
 * Proves: write entries, severity filter, subsystem filter, oldest-to-newest
 * order, ring overflow drops oldest, fail-closed on NULL. */
#include "wubu_klog.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_klog_t k;
    CHECK(wubu_klog_init(&k) == 0, "init ok");

    wubu_klog_write(&k, WUBU_KLOG_DEBUG, "mm", "alloc");
    wubu_klog_write(&k, WUBU_KLOG_INFO,  "sched", "switch");
    wubu_klog_write(&k, WUBU_KLOG_WARN,  "mm", "high water");
    wubu_klog_write(&k, WUBU_KLOG_ERROR, "net", "link down");
    CHECK(k.n == 4, "4 entries written");

    const wubu_klog_entry_t *out[16];

    /* severity filter: >= WARN -> 2 entries (warn, error) */
    int n = wubu_klog_query(&k, WUBU_KLOG_WARN, NULL, out, 16);
    CHECK(n == 2, ">= WARN -> 2 entries");
    CHECK(out[0]->level == WUBU_KLOG_WARN && out[1]->level == WUBU_KLOG_ERROR, "ordered warn then error");

    /* severity filter: >= INFO -> 3 (info, warn, error) */
    CHECK(wubu_klog_query(&k, WUBU_KLOG_INFO, NULL, out, 16) == 3, ">= INFO -> 3");

    /* all levels -> 4 */
    CHECK(wubu_klog_query(&k, WUBU_KLOG_DEBUG, NULL, out, 16) == 4, "all -> 4");

    /* subsystem filter: mm -> 2 (alloc debug, high water warn) */
    n = wubu_klog_query(&k, WUBU_KLOG_DEBUG, "mm", out, 16);
    CHECK(n == 2, "subsystem mm -> 2");
    CHECK(strcmp(out[0]->subsystem, "mm") == 0 && strcmp(out[1]->subsystem, "mm") == 0, "both mm");

    /* subsystem + severity: mm >= WARN -> 1 (high water) */
    CHECK(wubu_klog_query(&k, WUBU_KLOG_WARN, "mm", out, 16) == 1, "mm >= WARN -> 1");

    /* max_out limits emission */
    CHECK(wubu_klog_query(&k, WUBU_KLOG_DEBUG, NULL, out, 2) == 2, "max_out=2 caps emission");

    /* ring overflow: write past CAP, oldest dropped */
    wubu_klog_t big;
    wubu_klog_init(&big);
    for (int i = 0; i < WUBU_KLOG_CAP + 10; i++)
        wubu_klog_write(&big, WUBU_KLOG_INFO, "x", "msg");
    CHECK(big.n == WUBU_KLOG_CAP, "ring capped at CAP");
    /* oldest surviving is seq 11 (first 10 dropped) */
    wubu_klog_query(&big, WUBU_KLOG_DEBUG, NULL, out, 1);
    CHECK(out[0]->seq == 11, "oldest 10 dropped (first surviving seq 11)");

    /* fail-closed */
    CHECK(wubu_klog_query(&k, WUBU_KLOG_DEBUG, NULL, out, 0) == 0, "max_out=0 -> 0");
    CHECK(wubu_klog_query(NULL, WUBU_KLOG_DEBUG, NULL, out, 16) == 0, "NULL query -> 0");
    CHECK(wubu_klog_write(NULL, WUBU_KLOG_INFO, "x", "y") == 0, "NULL write -> 0");
    CHECK(wubu_klog_init(NULL) == -1, "NULL init -> -1");

    printf("\n=== test_klog: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
