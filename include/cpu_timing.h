#ifndef WUBU_CPU_TIMING_H
#define WUBU_CPU_TIMING_H

/**
 * cpu_timing.h — Cycle-accurate CPU timing utilities
 *
 * Adapted from tailslayer (LaurieWired/tailslayer):
 *   rdtsc_lfence, rdtscp_lfence, clflush, mfence, pin_to_core,
 *   virt_to_phys, compute_channel, tsc_calibrate
 *
 * For use in profiling CUDA kernels, benchmarking CPU ops, and
 * detecting memory system behavior (cache misses, DRAM refresh).
 *
 * All functions are static inline — zero call overhead.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>
#include <fcntl.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Cycle-accurate timing (x86_64)
// ============================================================

/** Read timestamp counter with lfence serialization. Returns CPU cycles. */
static inline uint64_t wubu_rdtsc(void) {
#if (defined(__x86_64__) || defined(__i386__)) && !defined(__CUDACC__)
    uint64_t lo, hi;
    __asm__ __volatile__ ("lfence\nrdtsc" : "=a"(lo), "=d"(hi));
    return (hi << 32) | lo;
#else
    return 0;
#endif
}

/** Read timestamp counter and processor ID with lfence. Returns CPU cycles. */
static inline uint64_t wubu_rdtscp(void) {
#if (defined(__x86_64__) || defined(__i386__)) && !defined(__CUDACC__)
    uint64_t lo, hi;
    unsigned int aux;
    __asm__ __volatile__ ("lfence\nrdtscp" : "=a"(lo), "=d"(hi), "=c"(aux));
    return (hi << 32) | lo;
#else
    return 0;
#endif
}

/** Pin current thread to a specific CPU core. */
static inline int wubu_pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return sched_setaffinity(0, sizeof(cpuset), &cpuset);
}

// ============================================================
// Cache management
// ============================================================

/** Flush a single cache line containing address `addr`. */
static inline void wubu_clflush(const volatile void *addr) {
#if (defined(__x86_64__) || defined(__i386__)) && !defined(__CUDACC__)
    __asm__ __volatile__ ("clflush %0" :: "m"(*(const volatile char *)addr));
#endif
}

/** Flush range of memory. */
static inline void wubu_clflush_range(const volatile void *addr, size_t size) {
    const char *ptr = (const char *)addr;
    for (size_t i = 0; i < size; i += 64) {
        wubu_clflush(ptr + i);
    }
}

/** Memory barriers */
static inline void wubu_mfence(void) {
#if (defined(__x86_64__) || defined(__i386__)) && !defined(__CUDACC__)
    __asm__ __volatile__ ("mfence" ::: "memory");
#else
    __sync_synchronize();
#endif
}

/** Read TSC with serialization and return frequency estimate (cycles/sec). */
static inline double wubu_tsc_calibrate(int seconds) {
    uint64_t start = wubu_rdtsc();
    sleep(seconds);
    uint64_t end = wubu_rdtsc();
    return (double)(end - start) / seconds;
}

/** Virtual to physical address translation (Linux only, requires root). */
static inline uint64_t wubu_virt_to_phys(void *vaddr) {
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) return 0;
    
    uint64_t v = (uint64_t)vaddr;
    uint64_t page = v / 4096;
    uint64_t entry;
    if (lseek(fd, page * 8, SEEK_SET) == (off_t)-1) { close(fd); return 0; }
    if (read(fd, &entry, 8) != 8) { close(fd); return 0; }
    close(fd);
    
    if (!(entry & (1ULL << 63))) return 0; // Not present
    
    uint64_t pfn = entry & ((1ULL << 55) - 1);
    return (pfn * 4096) + (v % 4096);
}

#ifdef __cplusplus
}
#endif

#endif // WUBU_CPU_TIMING_H