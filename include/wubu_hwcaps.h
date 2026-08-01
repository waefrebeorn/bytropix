/*
 * wubu_hwcaps.h — Runtime hardware capability detection + SIMD ladder.
 *
 * Detects the CPU SIMD feature ladder (SSE4.2 -> AVX -> AVX2 -> AVX512F) plus
 * useful micro-architecture facts (L1/L2/L3 sizes guessed from CPUID leaf 0x4,
 * cache-line width). This is the single source of truth that the tandem engine,
 * Rambus KV bank, and GEMV auto-tuner all query so they dispatch to the widest
 * vector path the silicon supports — no compile-time -march guessing, no
 * SIGILL on an older core.
 *
 * Self-contained C11 + raw CPUID. No third-party deps.
 */
#ifndef WUBU_HWCAPS_H
#define WUBU_HWCAPS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_HW_SSE4_2  (1u << 0)
#define WUBU_HW_AVX     (1u << 1)
#define WUBU_HW_AVX2    (1u << 2)
#define WUBU_HW_AVX512F (1u << 3)
#define WUBU_HW_FMA     (1u << 4)
#define WUBU_HW_BMI2    (1u << 5)
#define WUBU_HW_HUGEPAGE (1u << 6)  /* kernel exposes transparent hugepages */

typedef struct {
    uint32_t flags;       /* WUBU_HW_* bitmask */
    int      simd_bits;   /* widest vector lane width: 128/256/512 */
    int      simd_lanes;  /* floats per widest lane: 4/8/16 */
    int      l1_bytes;
    int      l2_bytes;
    int      l3_bytes;
    int      cache_line;  /* bytes (typically 64) */
    int      family, model, stepping;
    char     brand[64];
} wubu_hwcaps_t;

/* Detect once. Cached; subsequent calls are free. */
const wubu_hwcaps_t *wubu_hwcaps_get(void);

/* True if the given flag bit is set. */
int wubu_hwcaps_has(uint32_t flag);

/* Human-readable summary into out (static storage). */
const char *wubu_hwcaps_str(const wubu_hwcaps_t *h);

/* Best SIMD suffix string for the current CPU ("avx512","avx2","avx","sse4"). */
const char *wubu_hwcaps_simd_name(const wubu_hwcaps_t *h);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_HWCAPS_H */
