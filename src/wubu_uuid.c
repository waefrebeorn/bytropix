/*
 * wubu_uuid.c — UUIDv7 generation (RFC 9562) for session tracking.
 *
 * Pure C11 port from WuBuOS/src/runtime/wubu_uuid.c.
 * Uses wubu_win.h for POSIX shims (clock_gettime, getpid, read) on Windows.
 *
 * SPDX-License-Identifier: Waefrebeorn-UMV3
 */
#include "wubu_uuid.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

/* Monotonic counter (64 bits for 48-bit timestamp + 16-bit sequence) */
static uint64_t g_uuid_seq = 0;

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

/* Format UUIDv7: 8-4-4-4-12 hex digits.
 *
 * UUIDv7 layout (RFC 9562):
 *   bytes 0-5:  timestamp_ms (48 bits)  → groups 1+2
 *   bytes 6-7:  version 7 (4b) + rand_a (12b) → group 3
 *   bytes 8-9:  variant (2b) + rand_a_top (14b) → group 4
 *   bytes 10-15: rand_b (48 bits) → group 5 (12 hex)
 *
 * We use a 64-bit monotonic sequence counter split across rand_a and rand_b.
 */
static void format_uuid(char *buf, uint64_t ts_ms, uint64_t seq) {
    uint32_t t_hi = (uint32_t)((ts_ms >> 16) & 0xFFFFFFFF);
    uint16_t t_lo = (uint16_t)(ts_ms & 0xFFFF);

    /* seq is split: low 12 bits → rand_a, high bits → rand_b */
    uint16_t rand_a = (uint16_t)(seq & 0xFFF);
    uint64_t rand_b = (seq >> 12) & 0xFFFFFFFFFFFFULL; /* 48 bits */

    uint16_t mid = (uint16_t)(0x7000 | (rand_a & 0x0FFF));  /* version 7 */
    uint16_t var_hi = (uint16_t)(0x8000 | ((rand_b >> 46) & 0x3FFF));
    uint32_t rand_lo = (uint32_t)(rand_b & 0xFFFFFFFF);
    uint16_t rand_mid = (uint16_t)((rand_b >> 32) & 0xFFFF);

    snprintf(buf, 37, "%08x-%04x-%04x-%04x-%04x%08x",
             t_hi, t_lo, mid, var_hi, rand_mid, rand_lo);
}

char *wubu_uuid_v7(char *buf, size_t len) {
    if (len < 37) return NULL;

    struct timespec ts;
    memset(&ts, 0, sizeof(ts));
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t ts_ms = (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;

    static int seeded = 0;
    if (!seeded) {
        g_uuid_seq = urandom_seed();
        seeded = 1;
    }

    g_uuid_seq++;
    format_uuid(buf, ts_ms, g_uuid_seq);
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
    /* UUID format: 8-4-4-4-12 → 6 groups */
    if (sscanf(uuid, "%8x-%4x-%4x-%4x-%4x%8x",
               &t_hi, &t_lo, &mid, &var_hi, &r_hi, &r_lo) != 6) return -1;

    *ver = (mid >> 12) & 0xF;
    *rand_a = mid & 0x0FFF;

    *ts_ms = ((int64_t)t_hi << 16) | (int64_t)t_lo;
    /* Reconstruct rand_b: high 14 bits from var_hi, low 48 from r_hi+r_lo */
    *rand_b = ((uint64_t)(var_hi & 0x3FFF) << 48) |
              ((uint64_t)r_hi << 32) | (uint64_t)r_lo;

    return 0;
}
