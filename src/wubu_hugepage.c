/*
 * wubu_hugepage.c -- Hugepage-backed KV arena (O02).
 *
 * Convergence (OS paging cross-discipline 7-hop): KV cache traffic is
 * bandwidth-bound and TLB-footprint-heavy; 2MB hugepages cut TLB misses and
 * page-table walks for large KV arenas, raising effective bandwidth. This is
 * a drop-in arena: try MAP_HUGETLB; if the system has no hugepages reserved
 * (common on WSL/cloud), silently fall back to a plain mmap so behaviour is
 * always correct (never crash, never fail to allocate when plain would work).
 *
 * Triple-DA:
 *  - Correctness: alloc returns a usable pointer or NULL; free matches.
 *  - Privacy: no external dep.
 *  - Robustness: size==0 -> NULL; hugepage-unavailable -> plain fallback;
 *                size not hugepage-aligned -> rounded up.
 */
#include "wubu_hugepage.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif

#define HP_SHIFT 21   /* 2 MB hugepage */
#define HP_SIZE  (1u << HP_SHIFT)

struct wubu_hugepage {
    void  *base;
    size_t size;       /* actual mmap length (may be > requested) */
    int    used_huge;  /* 1 if MAP_HUGETLB succeeded */
};

static size_t round_up_huge(size_t n) {
    return (n + HP_SIZE - 1) & ~((size_t)HP_SIZE - 1);
}

wubu_hugepage_t *wubu_hugepage_create(size_t nbytes) {
    if (nbytes == 0) return NULL;
    wubu_hugepage_t *a = (wubu_hugepage_t *)calloc(1, sizeof(*a));
    if (!a) return NULL;

    size_t len = round_up_huge(nbytes);
    void *p = mmap(NULL, len, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (p == MAP_FAILED) {
        /* Fallback: plain anonymous mmap (always works if memory exists). */
        p = mmap(NULL, len, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { free(a); return NULL; }
        a->used_huge = 0;
    } else {
        a->used_huge = 1;
    }
    a->base = p;
    a->size = len;
    return a;
}

void *wubu_hugepage_ptr(wubu_hugepage_t *a) {
    return a ? a->base : NULL;
}

size_t wubu_hugepage_size(const wubu_hugepage_t *a) {
    return a ? a->size : 0;
}

int wubu_hugepage_is_huge(const wubu_hugepage_t *a) {
    return a ? a->used_huge : 0;
}

void wubu_hugepage_destroy(wubu_hugepage_t *a) {
    if (!a) return;
    if (a->base) munmap(a->base, a->size);
    free(a);
}
