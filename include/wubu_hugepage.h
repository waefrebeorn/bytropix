/*
 * wubu_hugepage.h -- Hugepage-backed KV arena (O02). Opaque struct.
 */
#ifndef WUBU_HUGEPAGE_H
#define WUBU_HUGEPAGE_H

#include <stddef.h>

typedef struct wubu_hugepage wubu_hugepage_t;

/* Allocate an arena of at least `nbytes` (rounded up to 2MB). Tries hugepages;
 * falls back to plain mmap if unavailable. NULL on hard failure. */
wubu_hugepage_t *wubu_hugepage_create(size_t nbytes);

/* Base pointer (usable as a KV buffer). */
void *wubu_hugepage_ptr(wubu_hugepage_t *a);

/* Actual mapped length (>= requested). */
size_t wubu_hugepage_size(const wubu_hugepage_t *a);

/* 1 if backed by real hugepages, 0 if plain-mmap fallback. */
int wubu_hugepage_is_huge(const wubu_hugepage_t *a);

void wubu_hugepage_destroy(wubu_hugepage_t *a);

#endif /* WUBU_HUGEPAGE_H */
