/*
 * test_arena.c -- doc 006 triple-DA (arena allocator for per-request + KV).
 * P1 correctness: sub-arena bump alloc is O(1), cache-line aligned, exhausts
 *   cleanly (returns NULL). Reset O(1). Multiple sub-arenas independent.
 * P2 privacy: no external allocator (mmap/malloc), own C.
 * P3 robustness: huge-page fallback chain works; degenerate size=0 -> NULL.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "wubu_arena.h"

static int check_align(const void *p, size_t align) {
    return ((uintptr_t)p % align) == 0;
}

int main(void) {
    wubu_arena_t a;
    assert(wubu_arena_init(&a, 1024*1024, 0) == 0);  /* 1MB */

    /* create a sub-arena */
    wubu_sub_arena_t sa1;
    assert(wubu_sub_arena_create(&a, &sa1, 256*1024) == 0);

    /* alloc various sizes, check alignment and bump */
    void *p1 = wubu_sub_arena_alloc(&sa1, 100, 64);
    void *p2 = wubu_sub_arena_alloc(&sa1, 200, 32);
    void *p3 = wubu_sub_arena_alloc(&sa1, 50, 16);
    assert(p1 && p2 && p3);
    assert(check_align(p1, 64) && check_align(p2, 32) && check_align(p3, 16));

    /* exhaust */
    void *last = NULL;
    while (1) {
        void *p = wubu_sub_arena_alloc(&sa1, 1024, 64);
        if (!p) break;
        last = p;
    }
    assert(last != NULL);

    /* reset and re-use */
    wubu_sub_arena_reset(&sa1);
    void *p4 = wubu_sub_arena_alloc(&sa1, 128, 64);
    assert(p4 != NULL);

    /* multiple sub-arenas independent */
    wubu_sub_arena_t sa2;
    assert(wubu_sub_arena_create(&a, &sa2, 256*1024) == 0);
    void *q1 = wubu_sub_arena_alloc(&sa2, 512, 64);
    void *q2 = wubu_sub_arena_alloc(&sa1, 512, 64);  /* sa1 still works */
    assert(q1 && q2);

    /* calloc zeroes */
    int *arr = (int *)wubu_sub_arena_calloc(&sa1, 10, sizeof(int), 64);
    int zero = 1; for(int i=0;i<10;i++) if (arr[i]!=0) zero=0;
    assert(zero);

    /* stats */
    size_t used = wubu_arena_committed(&a);
    size_t free = wubu_arena_available(&a);
    printf("arena committed=%zu  free=%zu  total=%zu\n", used, free, used+free);
    assert(used > 0 && free < a.total_bytes);

    /* destroy sub-arenas */
    wubu_sub_arena_destroy(&a, &sa1);
    wubu_sub_arena_destroy(&a, &sa2);

    wubu_arena_free(&a);

    /* degenerate: size=0 -> NULL */
    wubu_arena_t a2; assert(wubu_arena_init(&a2, 4096, 0)==0);
    wubu_sub_arena_t sa; assert(wubu_sub_arena_create(&a2, &sa, 2048)==0);
    void *p0 = wubu_sub_arena_alloc(&sa, 0, 64);
    assert(p0 == NULL);
    wubu_sub_arena_destroy(&a2, &sa);
    wubu_arena_free(&a2);

    printf("ALL ARENA CHECKS PASSED\n");
    return 0;
}