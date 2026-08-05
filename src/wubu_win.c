/*
 * wubu_win.c — implementation of the Windows POSIX shims declared in wubu_win.h.
 * Linked into the Windows build of wubuwizard. Self-contained, no god headers.
 * SPDX-License-Identifier: WaefreBeorn-UMV3
 */
#if defined(_WIN32)

#include "wubu_win.h"
#include <windows.h>
#include <process.h>
#include <io.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- mmap / munmap / msync -------------------------------------------- *
 * We keep a small process-wide table mapping a returned base address back to
 * the bookkeeping needed to release it. Anonymous mappings use VirtualAlloc
 * with a prefix header; file-backed mappings use CreateFileMapping + a
 * registered entry (we return the mapped view directly). */

#define WUBU_MM_MAGIC 0x57424D01u
#define WUBU_MM_MAX   1024

typedef struct wubu_mm_node {
    void  *base;     /* address returned to caller */
    int    is_file;
    HANDLE map;      /* file mapping handle (file-backed) */
    struct wubu_mm_node *next;
} wubu_mm_node;

static wubu_mm_node g_mm_slots[WUBU_MM_MAX];
static wubu_mm_node *g_mm_free;
static wubu_mm_node *g_mm_used;
static volatile LONG g_mm_lock;

static wubu_mm_node *mm_alloc_node(void) {
    wubu_mm_node *n = g_mm_free;
    if (n) { g_mm_free = n->next; }
    else {
        for (int i = 0; i < WUBU_MM_MAX; i++) {
            if (g_mm_slots[i].base == NULL && g_mm_slots[i].map == NULL
                && g_mm_slots[i].next == NULL) {
                n = &g_mm_slots[i]; break;
            }
        }
    }
    if (n) { n->next = NULL; n->is_file = 0; n->map = NULL; }
    return n;
}
static void mm_lock(void)   { while (InterlockedExchange(&g_mm_lock, 1) == 1); }
static void mm_unlock(void) { InterlockedExchange(&g_mm_lock, 0); }

void *wubu_mmap(void *addr, size_t len, int prot, int flags, int fd, long off) {
    (void)prot; (void)addr;
    if (len == 0) return MAP_FAILED;

    if ((flags & MAP_ANONYMOUS) || fd < 0) {
        size_t total = len + sizeof(WUBU_MM_MAGIC);
        void *raw = VirtualAlloc(NULL, total, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!raw) return MAP_FAILED;
        *(unsigned *)raw = WUBU_MM_MAGIC;
        return (char *)raw + sizeof(WUBU_MM_MAGIC);
    }

    /* File-backed (MAP_SHARED or MAP_PRIVATE).
       Honor PROT_*: a read-only file (fopen "rb") cannot get a READWRITE
       mapping, so use PAGE_READONLY/FILE_MAP_READ unless PROT_WRITE is set. */
    int want_write = (prot & PROT_WRITE) != 0;
    DWORD page_prot = want_write ? PAGE_READWRITE : PAGE_READONLY;
    DWORD map_access = want_write ? FILE_MAP_WRITE : FILE_MAP_READ;
    HANDLE fh = (HANDLE)_get_osfhandle(fd);
    if (fh == INVALID_HANDLE_VALUE) return MAP_FAILED;
    HANDLE map = CreateFileMappingA(fh, NULL, page_prot, 0, 0, NULL);
    if (!map) return MAP_FAILED;
    void *view = MapViewOfFile(map, map_access, 0, (DWORD)off, len);
    if (!view) {
        DWORD le = GetLastError();
        fprintf(stderr, "wubu_mmap: MapViewOfFile failed (GetLastError=%lu, off=%ld, len=%zu)\n",
                (unsigned long)le, (long)off, (size_t)len);
        CloseHandle(map); return MAP_FAILED;
    }

    mm_lock();
    wubu_mm_node *n = mm_alloc_node();
    if (!n) { mm_unlock(); UnmapViewOfFile(view); CloseHandle(map); return MAP_FAILED; }
    n->base = view; n->is_file = 1; n->map = map; n->next = g_mm_used; g_mm_used = n;
    mm_unlock();
    return view;
}

int wubu_munmap(void *addr, size_t len) {
    (void)len;
    if (!addr) return -1;
    /* File-backed: find in table FIRST. For file maps `addr` is the raw
     * view base (no magic prefix), so we must NOT deref memory before it. */
    mm_lock();
    wubu_mm_node **pp = &g_mm_used;
    while (*pp) {
        if ((*pp)->base == addr) {
            wubu_mm_node *n = *pp;
            *pp = n->next;
            n->next = g_mm_free; g_mm_free = n;
            n->base = NULL; n->is_file = 0;
            mm_unlock();
            UnmapViewOfFile(addr);
            if (n->map) CloseHandle(n->map);
            n->map = NULL;
            return 0;
        }
        pp = &(*pp)->next;
    }
    mm_unlock();
    /* Not file-backed: must be an anonymous allocation with a magic prefix. */
    unsigned *pfx = (unsigned *)((char *)addr - sizeof(WUBU_MM_MAGIC));
    if (*pfx == WUBU_MM_MAGIC) {
        VirtualFree(pfx, 0, MEM_RELEASE);
        return 0;
    }
    return -1;
}

int wubu_msync(void *addr, size_t len, int flags) {
    (void)flags;
    return FlushViewOfFile(addr, len) ? 0 : -1;
}

/* NOTE: the shell-free launcher (fork/exec) is provided by wubu_spawn_win.c
 * (CreateProcess-based) and routed via wubu_spawn.c's _WIN32 branch. This
 * file only supplies the mmap/sched/sysconf/posix_memalign/clock_gettime/
 * setenv shims the engine needs on MSYS2. */

#endif /* _WIN32 */
