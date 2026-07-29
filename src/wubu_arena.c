/*
 * wubu_arena.c -- Arena allocator for per-request + KV buffers (doc 006).
 * Self-contained C11. See header.
 */
#include "wubu_arena.h"
#include <stdlib.h>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>

/* Round up to power-of-two alignment */
static size_t align_up(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

/* Try mmap with huge pages; fallback to regular mmap; final fallback malloc */
static void *try_mmap(size_t size, int huge, size_t *actual) {
#if defined(__linux__) || defined(__APPLE__)
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    int prot = PROT_READ | PROT_WRITE;
    if (huge) {
#if defined(MAP_HUGETLB) && defined(MAP_HUGE_2MB)
        flags |= MAP_HUGETLB | MAP_HUGE_2MB;
#endif
    }
    void *p = mmap(NULL, size, prot, flags, -1, 0);
    if (p != MAP_FAILED) { *actual = size; return p; }
    /* fallback: regular mmap */
    flags = MAP_PRIVATE | MAP_ANONYMOUS;
    p = mmap(NULL, size, prot, flags, -1, 0);
    if (p != MAP_FAILED) { *actual = size; return p; }
#endif
    /* final fallback: malloc + align */
    void *raw = malloc(size + WUBU_ARENA_PAGE);
    if (!raw) return NULL;
    uintptr_t r = (uintptr_t)raw;
    uintptr_t aligned = align_up(r, WUBU_ARENA_PAGE);
    *actual = size;
    return (void *)aligned;
}

int wubu_arena_init(wubu_arena_t *a, size_t total_bytes, int use_huge_pages) {
    if (!a || total_bytes == 0) return -1;
    memset(a, 0, sizeof(*a));
    size_t actual;
    a->base = try_mmap(total_bytes, use_huge_pages, &actual);
    if (!a->base) return -1;
    a->limit = a->base + actual;
    a->total_bytes = actual;
    a->huge_pages = use_huge_pages;
    return 0;
}

void wubu_arena_free(wubu_arena_t *a) {
    if (!a || !a->base) return;
    munmap(a->base, a->total_bytes);
    a->base = a->limit = NULL;
    a->total_bytes = a->used_bytes = 0;
}

int wubu_sub_arena_create(wubu_arena_t *a, wubu_sub_arena_t *out, size_t bytes) {
    if (!a || !out || bytes == 0) return -1;
    /* Align slice start to cache line, size to page for clean boundaries */
    size_t slice_bytes = align_up(bytes, WUBU_ARENA_PAGE);
    size_t align = WUBU_ARENA_CACHELINE;
    size_t start = align_up((size_t)(a->limit - a->base) - a->used_bytes - slice_bytes, align);
    if (start + slice_bytes > a->total_bytes) return -1; /* out of space */
    /* Actually we bump from the END backwards so sub-arenas stack down.
     * But simpler: bump from current used_bytes forward. */
    size_t bump = align_up(a->used_bytes, align);
    if (bump + slice_bytes > a->total_bytes) return -1;
    out->base = a->base + bump;
    out->bump = out->base;
    out->limit = out->base + slice_bytes;
    out->used = 0;
    a->used_bytes = bump + slice_bytes;
    return 0;
}

void wubu_sub_arena_reset(wubu_sub_arena_t *sa) {
    if (!sa) return;
    sa->bump = sa->base;
    sa->used = 0;
}

void wubu_sub_arena_destroy(wubu_arena_t *a, wubu_sub_arena_t *sa) {
    (void)a; (void)sa;
    /* For simplicity: we don't return slices to a free list.
     * In production: add to free_list for reuse. Here we just reset. */
    wubu_sub_arena_reset(sa);
}

void *wubu_sub_arena_alloc(wubu_sub_arena_t *sa, size_t size, size_t align) {
    if (!sa || size == 0) return NULL;
    if (align == 0) align = WUBU_ARENA_CACHELINE;
    size_t offset = (size_t)(sa->bump - sa->base);
    size_t aligned = align_up(offset, align);
    if (aligned + size > (size_t)(sa->limit - sa->base)) return NULL;
    void *p = sa->base + aligned;
    sa->bump = (uint8_t *)p + size;
    sa->used = (size_t)(sa->bump - sa->base);
    return p;
}

void *wubu_sub_arena_calloc(wubu_sub_arena_t *sa, size_t nmemb, size_t size, size_t align) {
    size_t total = nmemb * size;
    void *p = wubu_sub_arena_alloc(sa, total, align);
    if (p) memset(p, 0, total);
    return p;
}

size_t wubu_arena_committed(const wubu_arena_t *a) { return a ? a->used_bytes : 0; }
size_t wubu_arena_available(const wubu_arena_t *a) { return a ? a->total_bytes - a->used_bytes : 0; }