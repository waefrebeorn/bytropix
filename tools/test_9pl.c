/* test_9pl.c -- NAMESPACE #194 9P2000.L openat + renameat.
 * Proves: openat relative to a dir fid, renameat moves a node across dirs,
 * rename rejects taken names / missing source / cycles, fail-closed on NULL. */
#include "wubu_9pl.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_9pl_t t;
    CHECK(wubu_9pl_init(&t) == 0, "init ok");
    /* /a (dir), /a/file1, /b (dir), /b/file2 */
    int a = wubu_9pl_mkdir(&t, -1, "a");
    int f1 = wubu_9pl_mkdir(&t, a, "file1");
    int b = wubu_9pl_mkdir(&t, -1, "b");
    int f2 = wubu_9pl_mkdir(&t, b, "file2");
    CHECK(t.n == 4, "4 nodes");

    /* openat: open 'file1' relative to dir a */
    CHECK(wubu_9pl_openat(&t, a, "file1") == f1, "openat a/file1 -> f1");
    /* openat is dir-relative: 'file2' is NOT in a */
    CHECK(wubu_9pl_openat(&t, a, "file2") == -1, "openat a/file2 -> -1 (not in a)");
    CHECK(wubu_9pl_openat(&t, b, "file2") == f2, "openat b/file2 -> f2");
    /* openat with a bad dir */
    CHECK(wubu_9pl_openat(&t, 99, "x") == -1, "openat bad dir -> -1");

    /* renameat: move file1 from a to b, renamed 'moved' */
    CHECK(wubu_9pl_renameat(&t, a, "file1", b, "moved") == 0, "renameat a/file1 -> b/moved");
    /* file1 no longer in a */
    CHECK(wubu_9pl_openat(&t, a, "file1") == -1, "file1 gone from a");
    /* now findable as b/moved */
    int moved = wubu_9pl_openat(&t, b, "moved");
    CHECK(moved == f1, "openat b/moved -> f1 (same node, moved)");
    CHECK(t.nodes[f1].parent == b, "f1 parent is now b");

    /* renameat onto a taken name fails */
    CHECK(wubu_9pl_renameat(&t, b, "moved", b, "file2") == -1, "rename onto taken name -> -1");
    /* renameat of a missing source fails */
    CHECK(wubu_9pl_renameat(&t, a, "nope", b, "x") == -1, "rename missing source -> -1");

    /* cycle prevention: can't move dir a into its own child */
    int sub = wubu_9pl_mkdir(&t, a, "sub");
    CHECK(wubu_9pl_renameat(&t, -1, "a", sub, "a2") == -1, "move dir into own subtree -> -1");

    /* fail-closed on NULL */
    CHECK(wubu_9pl_mkdir(NULL, -1, "x") == -1, "NULL mkdir -> -1");
    CHECK(wubu_9pl_openat(NULL, 0, "x") == -1, "NULL openat -> -1");
    CHECK(wubu_9pl_renameat(NULL, 0, "a", 0, "b") == -1, "NULL renameat -> -1");
    CHECK(wubu_9pl_init(NULL) == -1, "NULL init -> -1");

    printf("\n=== test_9pl: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
