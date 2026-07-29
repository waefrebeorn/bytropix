/*
 * wubu_arena.h -- Arena allocator for per-request + KV buffers (doc 006 / I01/C01).
 *
 * WHY (Kevin-Bacon convergence): game-console discipline (N64/PS2/GC era) +
 * modern inference. An arena allocator gives:
 * - Bounded peak memory: all allocations from a fixed block (13GB cap on this box).
 * - O(1) alloc/free: bump-pointer (or per-thread bump + central free-list).
 * - Deterministic latency: no malloc/free syscalls in the hot path.
 * - Leak-proof by construction: destroy arena = free everything, no per-object
 *   tracking. Matches the "no third-party" rule: our own C11, ~150 lines.
 *
 * DESIGN: One global arena for the process (pinned huge pages if available),
 * plus per-request sub-arenas carved from it. Sub-arenas support aligned
 * allocations for SIMD/GEMV buffers, KV pages (cache-line aligned), and
 * temporary scratch. Sub-arena destruction = reset its bump pointer.
 */
#ifndef WUBU_ARENA_H
#define WUBU_ARENA_H

#include <stdint.h>
#include <stddef.h>

/* Alignment constants (cache line = 64, SIMD = 64, page = 4096) */
#define WUBU_ARENA_CACHELINE 64
#define WUBU_ARENA_PAGE 4096

/* A sub-arena: a slice of the global arena with its own bump pointer.
 * Multiple threads can have their own sub-arena from the same global pool. */
typedef struct wubu_sub_arena {
    uint8_t *base;      /* start of this sub-arena's slice */
    uint8_t *bump;      /* next free byte */
    uint8_t *limit;     /* end of slice (exclusive) */
    size_t  used;       /* bytes used since last reset */
} wubu_sub_arena_t;

/* The global arena: fixed block allocated at init (supports huge pages). */
typedef struct {
    uint8_t *base;          /* start of global block */
    uint8_t *limit;         /* end of global block (exclusive) */
    size_t   total_bytes;   /* size of global block */
    size_t   used_bytes;    /* total committed across all sub-arenas */
    /* Optional: simple free list for returning sub-arena slices to pool */
    struct wubu_sub_arena *free_list;
    int huge_pages;         /* 1 if allocated with huge pages (2MB/1GB) */
} wubu_arena_t;

/* Initialize the global arena: allocates `total_bytes` (aligned to page).
 * If `use_huge_pages` and the platform supports it, tries mmap with
 * MAP_HUGETLB. Returns 0 on success. */
int wubu_arena_init(wubu_arena_t *a, size_t total_bytes, int use_huge_pages);
void wubu_arena_free(wubu_arena_t *a);

/* Carve a sub-arena from the global pool. `bytes` is the sub-arena's slice
 * size. Returns 0 on success, -1 if global arena exhausted. The caller owns
 * the returned `out` and must call `wubu_sub_arena_destroy` when done. */
int wubu_sub_arena_create(wubu_arena_t *a, wubu_sub_arena_t *out, size_t bytes);

/* Reset a sub-arena (free all its allocations in O(1)). Does NOT return
 * the slice to the global pool (caller decides when to destroy). */
void wubu_sub_arena_reset(wubu_sub_arena_t *sa);

/* Destroy a sub-arena: returns its slice to the global arena's free list
 * (or resets the global bump if it was the last allocation). */
void wubu_sub_arena_destroy(wubu_arena_t *a, wubu_sub_arena_t *sa);

/* Allocate `size` bytes with `align` (power of 2) from a sub-arena.
 * Returns NULL if the sub-arena is exhausted. O(1) bump-pointer. */
void *wubu_sub_arena_alloc(wubu_sub_arena_t *sa, size_t size, size_t align);

/* Allocate with zero-initialization. */
void *wubu_sub_arena_calloc(wubu_sub_arena_t *sa, size_t nmemb, size_t size, size_t align);

/* Stats: total committed / available in global arena. */
size_t wubu_arena_committed(const wubu_arena_t *a);
size_t wubu_arena_available(const wubu_arena_t *a);

#endif /* WUBU_ARENA_H */