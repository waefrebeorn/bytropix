/* test_9pf.c -- NAMESPACE #190 namespace is queryable (find /n -name).
 * Proves: tree build, full path, glob match, find by exact/glob, subtree
 * scoping, fail-closed on NULL. */
#include "wubu_9pf.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_9pf_t t;
    CHECK(wubu_9pf_init(&t) == 0, "init ok");
    /* build: /n
     *        |- kernel
     *        |   |- sched
     *        |   |- klog
     *        |- svc
     *        |   |- sched   (a different 'sched' under svc)
     */
    int n     = wubu_9pf_add(&t, -1, "n");
    int kern  = wubu_9pf_add(&t, n, "kernel");
    int sched = wubu_9pf_add(&t, kern, "sched");
    int klog  = wubu_9pf_add(&t, kern, "klog");
    int svc   = wubu_9pf_add(&t, n, "svc");
    int sched2= wubu_9pf_add(&t, svc, "sched");
    CHECK(t.n == 6, "6 nodes");

    /* full paths */
    char path[WUBU_9PF_PATHLEN];
    CHECK(wubu_9pf_path(&t, sched, path, sizeof path) > 0, "path built");
    CHECK(strcmp(path, "/n/kernel/sched") == 0, "sched path = /n/kernel/sched");
    wubu_9pf_path(&t, sched2, path, sizeof path);
    CHECK(strcmp(path, "/n/svc/sched") == 0, "svc/sched path = /n/svc/sched");

    /* glob matching */
    CHECK(wubu_9pf_glob("sched", "sched") == 1, "exact match");
    CHECK(wubu_9pf_glob("sched", "klog") == 0, "no match");
    CHECK(wubu_9pf_glob("s*", "sched") == 1, "s* matches sched");
    CHECK(wubu_9pf_glob("k*", "sched") == 0, "k* doesn't match sched");
    CHECK(wubu_9pf_glob("sc?ed", "sched") == 1, "? matches one char");
    CHECK(wubu_9pf_glob("*", "anything") == 1, "* matches all");

    /* find exact name 'sched' -> 2 matches (kernel + svc) */
    int ids[16];
    int cnt = wubu_9pf_find(&t, "sched", -1, ids, 16);
    CHECK(cnt == 2, "find 'sched' -> 2 matches");

    /* find scoped to kernel subtree -> 1 sched */
    cnt = wubu_9pf_find(&t, "sched", kern, ids, 16);
    CHECK(cnt == 1 && ids[0] == sched, "find 'sched' under kernel -> 1 (kernel/sched)");

    /* find glob 'k*' -> kernel, klog = 2 */
    cnt = wubu_9pf_find(&t, "k*", -1, ids, 16);
    CHECK(cnt == 2, "find 'k*' -> 2 (kernel, klog)");

    /* find under svc for 'k*' -> 0 */
    CHECK(wubu_9pf_find(&t, "k*", svc, ids, 16) == 0, "find 'k*' under svc -> 0");

    /* max caps the output (but count still reflects total) */
    cnt = wubu_9pf_find(&t, "sched", -1, ids, 1);
    CHECK(cnt == 2, "count reflects total even when max caps output");

    /* fail-closed */
    CHECK(wubu_9pf_add(&t, 99, "x") == -1, "bad parent -> -1");
    CHECK(wubu_9pf_add(&t, n, NULL) == -1, "NULL name -> -1");
    CHECK(wubu_9pf_path(&t, 99, path, sizeof path) == -1, "bad id path -> -1");
    CHECK(wubu_9pf_find(&t, "x", -1, NULL, 16) == 0, "NULL out -> 0");
    CHECK(wubu_9pf_glob(NULL, "x") == 0, "NULL glob -> 0");
    CHECK(wubu_9pf_init(NULL) == -1, "NULL init -> -1");

    printf("\n=== test_9pf: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
