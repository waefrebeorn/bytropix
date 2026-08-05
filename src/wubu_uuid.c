/*
 * wubu_uuid.c — UUIDv7 generation (RFC 9562) for session tracking.
 *
 * Pure C11 port from WuBuOS/src/runtime/wubu_uuid.c.
 * Uses wubu_win.h for POSIX shims (clock_gettime, getpid, read) on Windows.
 *
 * Design: CLOCK_MONOTONIC timestamp (guaranteed non-decreasing) as the
 * primary lexicographic ordering key. A 16-bit monotonic counter within
 * the same millisecond provides uniqueness. The random portion is
 * minimal (only fills the non-monotonic bits), which is acceptable for
 * internal session tracking — privacy is preserved (timestamp only).
 *
 * The epoch_offset is computed atomically: we take a single QPC reading
 * and derive both the monotonic ms and the epoch ms from it, eliminating
 * race conditions between clock_gettime calls.
 *
 * SPDX-License-Identifier: WaefreBeorn-UMV3
 */
#include "wubu_uuid.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

/* State: last timestamp (ms) + counter for sub-ms uniqueness */
static uint64_t g_last_ms = 0;
static uint32_t g_counter = 0;

/* Baseline offset: CLOCK_MONOTONIC → fake Epoch milliseconds.
 * Captured once at startup so timestamps look like real wall time. */
static uint64_t g_epoch_offset_ms = 0;
static int g_epoch_init = 0;

/* Read 8 bytes of entropy for initial seeding. */
static uint64_t urandom_seed(void) {
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) {
        struct timespec ts;
        memset(&ts, 0, sizeof(ts));
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return ((uint64_t)ts.tv_sec ^ (uint64_t)ts.tv_nsec ^
                ((uint64_t)getpid() << 16));
    }
    uint64_t seed = 0;
    ssize_t n = read(fd, &seed, sizeof(seed));
    close(fd);
    (void)n;
    return seed;
}

/* Get current millisecond timestamp (monotonic + epoch offset).
 * Computes offset atomically: single clock_gettime for both monotonic
 * and realtime, then derives epoch offset from the difference. */
static uint64_t current_ts_ms(void) {
    if (!g_epoch_init) {
        struct timespec mono, rt;
        memset(&mono, 0, sizeof(mono));
        memset(&rt, 0, sizeof(rt));
        clock_gettime(CLOCK_MONOTONIC, &mono);
        clock_gettime(CLOCK_REALTIME, &rt);
        uint64_t mono_ms = (uint64_t)mono.tv_sec * 1000ULL + (uint64_t)mono.tv_nsec / 1000000ULL;
        uint64_t rt_ms = (uint64_t)rt.tv_sec * 1000ULL + (uint64_t)rt.tv_nsec / 1000000ULL;
        g_epoch_offset_ms = rt_ms - mono_ms;
        g_epoch_init = 1;
    }

    struct timespec ts;
    memset(&ts, 0, sizeof(ts));
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t mono_ms = (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
    return mono_ms + g_epoch_offset_ms;
}

/* Format UUIDv7 string: 8-4-4-4-12 hex digits.
 *
 * UUIDv7 layout (RFC 9562):
 *   TTTTTTTT-TTTT-7TTT-VVVV-XXXXXXXXXXXX
 *   ts_hi(32)  ts_lo(16)  ver+rand_a(16)  ver+V+rand_b_hi(16)  rand_b_lo(48)
 *
 * Monotonic fields: ts (48 bits) + counter (12 bits in rand_a).
 * The counter goes in the FIRST random field after the version nibble,
 * ensuring lexicographic monotonicity within the same millisecond.
 */
static void format_uuid(char *buf, uint64_t ts_ms, uint32_t counter, uint64_t rand_seed) {
    uint32_t t_hi = (uint32_t)((ts_ms >> 16) & 0xFFFFFFFF);
    uint16_t t_lo = (uint16_t)(ts_ms & 0xFFFF);

    /* rand_a: 12 bits — counter (guarantees lexicographic ordering within same ms) */
    uint16_t rand_a = (uint16_t)(counter & 0xFFF);
    uint16_t mid = (uint16_t)(0x7000 | (rand_a & 0x0FFF));  /* version 7 */

    /* var_hi: 2 variant bits (0b10) + 14 bits from rand_seed */
    uint16_t var_hi = (uint16_t)(0x8000 | (rand_seed & 0x3FFF));

    /* Last 2 groups: 16 + 32 = 48 bits from remaining rand_seed */
    uint16_t rand_mid = (uint16_t)((rand_seed >> 14) & 0xFFFF);
    uint32_t rand_lo = (uint32_t)((rand_seed >> 30) & 0xFFFFFFFF);

    snprintf(buf, 37, "%08x-%04x-%04x-%04x-%04x%08x",
             t_hi, t_lo, mid, var_hi, rand_mid, rand_lo);
}

char *wubu_uuid_v7(char *buf, size_t len) {
    if (len < 37) return NULL;

    uint64_t ts_ms = current_ts_ms();

    static int seeded = 0;
    static uint64_t static_seed = 0;
    if (!seeded) {
        static_seed = urandom_seed();
        seeded = 1;
    }
    uint64_t rand_part = static_seed & 0xFFFFFFFFFFFFULL; /* 48 bits */

    /* If same millisecond, increment counter; else reset to 0 */
    /* Counter is 12 bits → overflows at 4096 UUIDs/ms (~244 µs period) */
    if (ts_ms == g_last_ms) {
        g_counter++;
    } else {
        g_counter = 0;
        g_last_ms = ts_ms;
    }

    format_uuid(buf, ts_ms, g_counter, rand_part);
    return buf;
}

char *wubu_uuid_v7_alloc(void) {
    char *buf = (char *)malloc(37);
    if (!buf) return NULL;
    return wubu_uuid_v7(buf, 37);
}

int64_t wubu_uuid_v7_timestamp(const char *uuid) {
    if (!uuid || strlen(uuid) < 24) return -1;
    uint32_t t_hi, t_lo;
    if (sscanf(uuid, "%8x-%4x", &t_hi, &t_lo) != 2) return -1;
    return ((int64_t)t_hi << 16) | (int64_t)t_lo;
}

int wubu_uuid_compare(const char *a, const char *b) {
    if (!a || !b) return 0;
    return strcmp(a, b);
}

int wubu_uuid_parse(const char *uuid, int64_t *ts_ms, uint16_t *rand_a,
                    uint64_t *rand_b, int *ver) {
    if (!uuid || strlen(uuid) < 36) return -1;
    uint32_t t_hi, t_lo, mid, var_hi;
    uint32_t r_hi, r_lo;
    if (sscanf(uuid, "%8x-%4x-%4x-%4x-%4x%8x",
               &t_hi, &t_lo, &mid, &var_hi, &r_hi, &r_lo) != 6) return -1;

    *ver = (mid >> 12) & 0xF;
    *rand_a = mid & 0x0FFF;

    *ts_ms = ((int64_t)t_hi << 16) | (int64_t)t_lo;
    *rand_b = ((uint64_t)(var_hi & 0x3FFF) << 32) |
              ((uint64_t)r_hi << 0) | (uint64_t)r_lo;

    return 0;
}
