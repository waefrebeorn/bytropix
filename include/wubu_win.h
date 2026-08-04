/*
 * wubu_win.h — Windows (MSYS2/mingw64) port shims for the wubuwizard engine.
 *
 * The engine was written Linux-first. The MSYS2 mingw64 toolchain is a native
 * Windows target and lacks several POSIX headers (sys/mman.h, sched.h,
 * sys/wait.h), so we supply the missing pieces here. Standard types that DO
 * exist on MSYS2 (pid_t via sys/types.h, struct timespec via time.h) are NOT
 * redefined — we guard them.
 *
 * Each shim is provided only for what is genuinely missing; files pull this in
 * via the Makefile's -include so no per-file edits are needed. We implement:
 *   mmap/munmap/msync   (VirtualAlloc / CreateFileMapping)
 *   sched_setaffinity + cpu_set_t
 *   posix_memalign / setenv / unsetenv / clock_gettime / memmem (if absent)
 *   sysconf(_SC_NPROCESSORS_ONLN / _SC_PAGESIZE)
 *   sys/resource.h, sys/sysinfo.h, fnmatch.h live in include/win32/
 *
 * SPDX-License-Identifier: WaefreBeorn-UMV3
 */
#ifndef WUBU_WIN_H
#define WUBU_WIN_H

#if defined(_WIN32)

#include <windows.h>
#include <process.h>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/* ---- sys/mman.h equivalents ------------------------------------------- */
#ifndef _SYS_MMAN_H_COMPAT
#define PROT_READ   0x1
#define PROT_WRITE  0x2
#define MAP_PRIVATE 0x02
#define MAP_SHARED  0x01
#define MAP_ANONYMOUS 0x20
#define MAP_NORESERVE 0x04000   /* no-op on Windows (commit immediately) */
#define MAP_HUGETLB 0x40000     /* no-op on Windows (no hugepages) */
#define MAP_HUGE_2MB 0x08000
#define MAP_WRITE   0x02        /* alias (Windows views are RW) */
#define MAP_FAILED  ((void *)-1)
#define MS_SYNC     0x0004

void *wubu_mmap(void *addr, size_t len, int prot, int flags, int fd, long off);
int   wubu_munmap(void *addr, size_t len);
int   wubu_msync(void *addr, size_t len, int flags);

static inline void *mmap(void *addr, size_t len, int prot, int flags,
                         int fd, long off) {
    return wubu_mmap(addr, len, prot, flags, fd, off);
}
static inline int munmap(void *addr, size_t len) {
    return wubu_munmap(addr, len);
}
static inline int msync(void *addr, size_t len, int flags) {
    return wubu_msync(addr, len, flags);
}
#endif

/* ---- sched.h equivalents ---------------------------------------------- */
#ifndef _SCHED_H_COMPAT
typedef uintptr_t cpu_set_t;
#define CPU_ZERO(set)      (*(set) = 0)
#define CPU_SET(c, set)    (*(set) |= ((uintptr_t)1 << (c)))
#define CPU_SETSIZE        1024
static inline int wubu_sched_setaffinity(int pid, size_t sz, cpu_set_t *set) {
    (void)pid; (void)sz;
    DWORD_PTR mask = (DWORD_PTR)(*set);
    if (mask == 0) return 0;
    if (SetThreadAffinityMask(GetCurrentThread(), mask)) return 0;
    return -1;
}
#define sched_setaffinity(pid, sz, set) wubu_sched_setaffinity((pid), (sz), (set))
#endif

/* ---- sysconf (wubu_affinity.c, gguf_reader.c, etc.) ------------------ */
#ifndef _SC_NPROCESSORS_ONLN
#define _SC_NPROCESSORS_ONLN 1
#define _SC_PAGESIZE 2
static inline long wubu_sysconf(int name) {
    if (name == _SC_NPROCESSORS_ONLN) {
        SYSTEM_INFO si; GetSystemInfo(&si);
        return (long)si.dwNumberOfProcessors;
    }
    if (name == _SC_PAGESIZE) {
        /* Windows MapViewOfFile requires the mapping OFFSET to be a multiple of
           the allocation granularity (65536), NOT the 4096 page size. Return
           the allocation granularity so mmap-offset alignment is correct. */
        SYSTEM_INFO si; GetSystemInfo(&si);
        return (long)si.dwAllocationGranularity;
    }
    return 1;
}
#define sysconf(n) wubu_sysconf(n)
#endif

/* ---- posix_memalign (if MSYS2 lacks it) ------------------------------ */
#ifndef HAVE_POSIX_MEMALIGN
static inline int wubu_posix_memalign(void **memptr, size_t alignment, size_t size) {
    void *p = _aligned_malloc(size, alignment);
    if (!p) return 12; /* ENOMEM */
    *memptr = p;
    return 0;
}
#define posix_memalign(p, a, s) wubu_posix_memalign((p), (a), (s))
#endif

/* ---- clock_gettime (if MSYS2 lacks it) ------------------------------- */
#ifndef HAVE_CLOCK_GETTIME
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
struct wubu_timespec { long tv_sec; long tv_nsec; };
static inline int wubu_clock_gettime(int clk, struct wubu_timespec *ts) {
    (void)clk;
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    ts->tv_sec  = (long)(cnt.QuadPart / freq.QuadPart);
    ts->tv_nsec = (long)(((cnt.QuadPart % freq.QuadPart) * 1000000000) / freq.QuadPart);
    return 0;
}
#define clock_gettime(c, t) wubu_clock_gettime((c), (struct wubu_timespec *)(t))
#endif

/* ---- setenv / unsetenv (if MSYS2 lacks them) ------------------------- */
#ifndef HAVE_SETENV
static inline int wubu_setenv(const char *name, const char *value, int overwrite) {
    (void)overwrite;
    char buf[512];
    snprintf(buf, sizeof(buf), "%s=%s", name, value);
    return _putenv(buf);
}
#define setenv(n, v, o) wubu_setenv((n), (v), (o))
#endif
#ifndef HAVE_UNSETENV
static inline int wubu_unsetenv(const char *name) {
    char buf[512];
    snprintf(buf, sizeof(buf), "%s=", name);
    return _putenv(buf);
}
#define unsetenv(n) wubu_unsetenv((n))
#endif

/* ---- memmem (wubu_model_adapter.c) ------------------------------------ */
#ifndef HAVE_MEMMEM
static inline void *wubu_memmem(const void *haystack, size_t hlen,
                                const void *needle, size_t nlen) {
    if (nlen == 0 || hlen < nlen) return NULL;
    const char *h = (const char *)haystack;
    const char *n = (const char *)needle;
    for (size_t i = 0; i + nlen <= hlen; i++) {
        if (memcmp(h + i, n, nlen) == 0) return (void *)(h + i);
    }
    return NULL;
}
#define memmem(h, hl, n, nl) wubu_memmem((h), (hl), (n), (nl))
#endif

/* ---- sys/resource.h (wubu_moe_backward.c) ----------------------------- */
#ifndef _SYS_RESOURCE_H_COMPAT
#define _SYS_RESOURCE_H_COMPAT
struct rusage { long ru_utime; long ru_stime; long ru_maxrss; long ru_minflt;
                long ru_majflt; long ru_inblock; long ru_oublock; long ru_nvcsw;
                long ru_nivcsw; };
#define RUSAGE_SELF 0
static inline int getrusage(int who, struct rusage *r) {
    (void)who; if (r) memset(r, 0, sizeof(*r)); return 0;
}
#endif

/* ---- open() flag normalization (MSYS2 uses _O_* names) --------------- */
#ifndef O_CREAT
#define O_CREAT   _O_CREAT
#define O_RDWR    _O_RDWR
#define O_WRONLY  _O_WRONLY
#define O_RDONLY  _O_RDONLY
#endif

/* ---- Win32 spawn entry points (wubu_spawn_win.c) --------------------- */
#include "wubu_spawn.h"
int wubu_spawn_win_capture(const char *file, char *const argv[],
                           char *out_buf, size_t out_cap, int *out_exit);
int wubu_spawn_win_wait(const char *file, char *const argv[], int silent);

#endif /* _WIN32 */
#endif /* WUBU_WIN_H */
