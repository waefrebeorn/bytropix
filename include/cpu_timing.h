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

// For CPU_ZERO/CPU_SET on non-glibc systems. Must be before sched.h include.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

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
#if defined(__x86_64__) || defined(__i386__)
    uint64_t lo, hi;
    asm volatile("lfence\n\t"
                 "rdtsc"
                 : "=a"(lo), "=d"(hi));
    return (hi << 32) | lo;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

/** Read timestamp counter with rdtscp + lfence. Returns CPU cycles + AUX (core ID). */
static inline uint64_t wubu_rdtscp(uint32_t *aux) {
#if defined(__x86_64__) || defined(__i386__)
    uint64_t lo, hi;
    asm volatile("rdtscp"
                 : "=a"(lo), "=d"(hi), "=c"(*aux));
    asm volatile("lfence" ::: "memory");
    return (hi << 32) | lo;
#else
    if (aux) *aux = 0;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

/** Cache-line flush: evict addr from all cache levels. */
static inline void wubu_clflush(void *addr) {
#if defined(__x86_64__) || defined(__i386__)
    asm volatile("clflush (%0)" :: "r"(addr) : "memory");
#else
    (void)addr;
#endif
}

/** Memory fence: serialize all load/store instructions. */
static inline void wubu_mfence(void) {
#if defined(__x86_64__) || defined(__i386__)
    asm volatile("mfence" ::: "memory");
#else
    __sync_synchronize();
#endif
}

/** Load fence: serialize instruction execution. */
static inline void wubu_lfence(void) {
#if defined(__x86_64__) || defined(__i386__)
    asm volatile("lfence" ::: "memory");
#else
    __sync_synchronize();
#endif
}

// ============================================================
// TSC Calibration (nanosleep-based)
// ============================================================

/**
 * Calibrate TSC frequency in GHz.
 * Sleeps 100ms and measures TSC ticks elapsed.
 * Returns GHz (e.g., 3.0 for 3.0 GHz).
 */
static inline double wubu_tsc_calibrate_ghz(void) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    uint64_t tsc0 = wubu_rdtsc();

    struct timespec req = {0, 100000000}; // 100ms
    nanosleep(&req, NULL);

    uint32_t aux;
    uint64_t tsc1 = wubu_rdtscp(&aux);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed_ns = (double)(t1.tv_sec - t0.tv_sec) * 1e9 +
                        (double)(t1.tv_nsec - t0.tv_nsec);
    return (double)(tsc1 - tsc0) / elapsed_ns;
}

/**
 * Convert TSC cycles to nanoseconds.
 */
static inline double wubu_cycles_to_ns(uint64_t cycles, double tsc_ghz) {
    return (double)cycles / tsc_ghz;
}

/**
 * Convert TSC cycles to microseconds.
 */
static inline double wubu_cycles_to_us(uint64_t cycles, double tsc_ghz) {
    return (double)cycles / (tsc_ghz * 1000.0);
}

// ============================================================
// CPU Pinning
// ============================================================

/** Pin current thread to a specific core. Returns 0 on success. */
static inline int wubu_pin_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return sched_setaffinity(0, sizeof(cpuset), &cpuset);
}

// ============================================================
// Physical address resolution (for DRAM channel detection)
// ============================================================

/**
 * Resolve virtual address to physical address via /proc/self/pagemap.
 * Returns 0 on failure (requires root or CAP_SYS_ADMIN).
 */
static inline uint64_t wubu_virt_to_phys(uint64_t vaddr) {
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) return 0;
    uint64_t entry;
    off_t offset = (vaddr / 4096) * 8;
    if (pread(fd, &entry, 8, offset) != 8) { close(fd); return 0; }
    close(fd);
    if (!(entry & (1ULL << 63))) return 0; // not present
    uint64_t pfn = entry & ((1ULL << 55) - 1);
    return (pfn * 4096) | (vaddr & 0xFFF);
}

/**
 * Compute DRAM channel number from physical address.
 * Channel bit varies by platform: 6 for DDR4, 8 for DDR5 typical.
 * Returns 0 or 1 for dual-channel, or more for multi-channel.
 */
static inline int wubu_compute_channel(uint64_t phys_addr, int channel_bit) {
    return (int)((phys_addr >> channel_bit) & 1);
}

// ============================================================
// Benchmark helpers: timed section
// ============================================================

/** 
 * Begin a timed section. Returns start cycle count.
 * Call wubi_timing_end() after the section.
 */
static inline uint64_t wubu_timing_start(void) {
    wubu_lfence();
    return wubu_rdtsc();
}

/**
 * End a timed section. Returns elapsed cycles.
 * Computes delta = end_cycles - start_cycles.
 */
static inline uint64_t wubu_timing_end(uint64_t start) {
    uint32_t aux;
    uint64_t end = wubu_rdtscp(&aux);
    wubu_lfence();
    return end - start;
}

/**
 * Clflush+reload timing probe: measure read latency for a single address.
 * Returns cycles taken to reload after clflush.
 * High latency (> median*2) indicates DRAM refresh (tREFI) stall.
 */
static inline uint64_t wubu_cache_probe(volatile void *addr) {
    wubu_clflush((void*)addr);
    wubu_mfence();
    wubu_lfence();
    uint64_t t0 = wubu_rdtsc();
    (void)*(volatile char*)addr;  // force read
    return wubu_timing_end(t0);
}

// ============================================================
// Memory channel info
// ============================================================

/** Print system timing info to stderr */
static inline void wubu_print_timing_info(double tsc_ghz) {
    fprintf(stderr, "  TSC: %.3f GHz (%.2f cycles/ns)\n",
            tsc_ghz, tsc_ghz);
    fprintf(stderr, "  1 cycle = %.2f ns\n", 1.0 / tsc_ghz);
    fprintf(stderr, "  1 us = %.0f cycles\n", tsc_ghz * 1000.0);
    fprintf(stderr, "  1 ms = %.0f cycles\n", tsc_ghz * 1000000.0);
}

#ifdef __cplusplus
}
#endif

#endif // WUBU_CPU_TIMING_H
