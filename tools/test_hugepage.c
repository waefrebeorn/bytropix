/*
 * test_hugepage.c -- O02 Hugepage KV arena verification.
 * Verifies: alloc returns usable writable memory, fallback works (plain mmap
 * when no hugepages reserved), size rounded to 2MB, edge cases (0 -> NULL).
 */
#include "wubu_hugepage.h"
#include <stdio.h>
#include <string.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

int main(void) {
    printf("=== test_hugepage (O02 hugepage KV arena) ===\n");

    /* size 0 -> NULL */
    CHECK(wubu_hugepage_create(0) == NULL, "size 0 -> NULL");

    /* small alloc (rounded up to 2MB) */
    wubu_hugepage_t *a = wubu_hugepage_create(4096);
    CHECK(a != NULL, "alloc succeeds");
    if (a) {
        void *p = wubu_hugepage_ptr(a);
        CHECK(p != NULL, "base ptr non-null");
        CHECK(wubu_hugepage_size(a) >= 4096, "size >= requested");
        CHECK((wubu_hugepage_size(a) & ((1u << 21) - 1)) == 0, "size 2MB-aligned");
        /* writable + readable (would segfault if bogus mapping) */
        memset(p, 0xAB, 1024);
        unsigned char *q = (unsigned char *)p;
        CHECK(q[0] == 0xAB && q[1023] == 0xAB, "memory is RW");
        /* huge or fallback both acceptable; report which */
        printf("  (arena backed by %s pages)\n",
               wubu_hugepage_is_huge(a) ? "HUGE" : "plain-mmap-fallback");
        wubu_hugepage_destroy(a);
    }

    /* large alloc */
    wubu_hugepage_t *b = wubu_hugepage_create(8 * 1024 * 1024);
    CHECK(b != NULL, "8MB alloc succeeds");
    if (b) {
        CHECK(wubu_hugepage_size(b) >= 8 * 1024 * 1024, "8MB size honored");
        wubu_hugepage_destroy(b);
    }

    if (failures == 0) { printf("ALL HUGEPAGE TESTS PASSED\n"); return 0; }
    printf("%d HUGEPAGE TEST(S) FAILED\n", failures);
    return 1;
}
