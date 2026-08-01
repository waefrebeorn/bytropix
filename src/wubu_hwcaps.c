/*
 * wubu_hwcaps.c — Runtime hardware capability detection (CPUID ladder).
 * See header. Self-contained C11. Raw CPUID, no third-party deps.
 */
#include "wubu_hwcaps.h"
#include <string.h>
#include <stdio.h>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

static wubu_hwcaps_t g_caps;
static int g_done = 0;

static void cpuid(uint32_t leaf, uint32_t subleaf,
                  uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *d) {
#if defined(__x86_64__) || defined(__i386__)
    __cpuid_count(leaf, subleaf, *a, *b, *c, *d);
#else
    *a = *b = *c = *d = 0;
#endif
}

const wubu_hwcaps_t *wubu_hwcaps_get(void) {
    if (g_done) return &g_caps;
    memset(&g_caps, 0, sizeof(g_caps));
    g_caps.cache_line = 64;
    g_caps.l1_bytes = 32768;
    g_caps.l2_bytes = 262144;
    g_caps.l3_bytes = 0;

#if defined(__x86_64__) || defined(__i386__)
    uint32_t a, b, c, d;
    cpuid(1, 0, &a, &b, &c, &d);
    g_caps.family  = ((a >> 8) & 0xF) + ((a >> 20) & 0xFF);
    g_caps.model   = ((a >> 4) & 0xF) | (((a >> 16) & 0xF) << 4);
    g_caps.stepping = a & 0xF;
    if (c & (1u << 20)) g_caps.flags |= WUBU_HW_SSE4_2;
    if (c & (1u << 28)) g_caps.flags |= WUBU_HW_AVX;
    if (c & (1u << 12)) g_caps.flags |= WUBU_HW_FMA;

    cpuid(7, 0, &a, &b, &c, &d);
    if (b & (1u << 5))  g_caps.flags |= WUBU_HW_AVX2;
    if (b & (1u << 16)) g_caps.flags |= WUBU_HW_AVX512F;
    if (b & (1u << 8))  g_caps.flags |= WUBU_HW_BMI2;

    /* Brand string (leaves 0x80000002..0x80000004) */
    uint32_t brand[12];
    cpuid(0x80000002, 0, &brand[0],  &brand[1],  &brand[2],  &brand[3]);
    cpuid(0x80000003, 0, &brand[4],  &brand[5],  &brand[6],  &brand[7]);
    cpuid(0x80000004, 0, &brand[8],  &brand[9],  &brand[10], &brand[11]);
    memcpy(g_caps.brand, brand, sizeof(g_caps.brand));
    g_caps.brand[63] = '\0';

    /* Cache topology (leaf 0x4). Sum L3 across cores for an upper bound. */
    for (uint32_t i = 0; i < 4; i++) {
        cpuid(4, i, &a, &b, &c, &d);
        int cache_type = a & 0x1F;
        if (cache_type == 0) break;
        int level = (a >> 5) & 0x7;
        int ways     = ((b >> 22) & 0x3FF) + 1;
        int partitions = ((b >> 12) & 0x3FF) + 1;
        int line_sz  = (b & 0xFFF) + 1;
        int sets     = c + 1;
        int size = ways * partitions * line_sz * sets;
        if (level == 1) g_caps.l1_bytes = size;
        else if (level == 2) g_caps.l2_bytes = size;
        else if (level == 3) g_caps.l3_bytes += size;
        g_caps.cache_line = line_sz;
    }
#endif

    if (g_caps.flags & WUBU_HW_AVX512F) { g_caps.simd_bits = 512; g_caps.simd_lanes = 16; }
    else if (g_caps.flags & WUBU_HW_AVX2) { g_caps.simd_bits = 256; g_caps.simd_lanes = 8; }
    else if (g_caps.flags & WUBU_HW_AVX)  { g_caps.simd_bits = 128; g_caps.simd_lanes = 4; }
    else if (g_caps.flags & WUBU_HW_SSE4_2) { g_caps.simd_bits = 128; g_caps.simd_lanes = 4; }

    g_done = 1;
    return &g_caps;
}

int wubu_hwcaps_has(uint32_t flag) {
    return (wubu_hwcaps_get()->flags & flag) != 0;
}

const char *wubu_hwcaps_str(const wubu_hwcaps_t *h) {
    static char buf[256];
    snprintf(buf, sizeof(buf),
        "SIMD=%dbit lanes=%d L1=%dB L2=%dB L3=%dB cl=%d [%s]",
        h->simd_bits, h->simd_lanes, h->l1_bytes, h->l2_bytes,
        h->l3_bytes, h->cache_line,
        (h->flags & WUBU_HW_AVX512F) ? "avx512" :
        (h->flags & WUBU_HW_AVX2)    ? "avx2" :
        (h->flags & WUBU_HW_AVX)     ? "avx"  : "sse4");
    return buf;
}

const char *wubu_hwcaps_simd_name(const wubu_hwcaps_t *h) {
    if (h->flags & WUBU_HW_AVX512F) return "avx512";
    if (h->flags & WUBU_HW_AVX2)    return "avx2";
    if (h->flags & WUBU_HW_AVX)     return "avx";
    return "sse4";
}
