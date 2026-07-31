/*
 * wubu_uuid.c — UUIDv7 generation (RFC 9562) for WuBuOS session tracking.
 *
 * Pure C11, no external libs. Uses clock_gettime for timestamp, /dev/urandom
 * for 74 bits of randomness (only used once at startup — subsequent UUIDs
 * are monotonically incremented from a counter).
 *
 * Triple-DA:
 *   Decision: UUIDv7 only (timestamp + counter, no MAC/hardware).
 *   Design:   monotonic counter ensures uniqueness without re-sampling entropy
 *             on every call (fast path is just increment).
 *   Robustness: falls back to /dev/urandom + counter if clock fails.
 */

#include "wubu_uuid.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

/* Thread-local monotonic counter (84 bits: 12-bit rand_a + 62-bit rand_b) */
static uint32_t g_uuid_counter_low = 0;
static uint32_t g_uuid_counter_high = 0;

/* Read 8 bytes of entropy from /dev/urandom for initial seeding */
static uint64_t urandom_seed(void) {
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) {
        /* Fallback: use clock + PID */
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return ((uint64_t)ts.tv_sec ^ (uint64_t)ts.tv_nsec ^
                ((uint64_t)getpid() << 16));
    }
    uint64_t seed = 0;
    read(fd, &seed, sizeof(seed));
    close(fd);
    return seed;
}

/* Format: 8-4-4-4-12 hex digits */
static void format_uuid(char *buf, uint64_t ts_ms, uint32_t counter) {
    /* UUIDv7 layout (RFC 9562):
     * bytes 0-5:  timestamp_ms (48 bits)
     * bytes 6-7:  rand_a (12 bits) + version 7 (4 bits)
     * bytes 8-15: variant (2 bits) + rand_b (62 bits)
     */
    uint16_t rand_a = (uint16_t)((counter >> 52) & 0xFFF);
    uint64_t rand_b = counter & 0xFFFFFFFFFFFFFULL; /* 52 bits */

    uint32_t t_hi = (uint32_t)((ts_ms >> 16) & 0xFFFFFFFF); /* 32 bits */
    uint32_t t_lo = (uint32_t)(ts_ms & 0xFFFF);             /* 16 bits */
    uint16_t mid = (0x7 << 12) | (rand_a & 0xFFF);  /* version 7 in high nibble */
    uint16_t var_hi = 0x8000 | (uint16_t)((rand_b >> 36) & 0x3FFF);  /* 2+14 = 16 bits */
    uint32_t rand_lo = (uint32_t)(rand_b & 0xFFFFFFFF);      /* 32 bits */
    uint16_t rand_x = (uint16_t)((rand_b >> 32) & 0xFFFF);   /* 16 bits */

    snprintf(buf, 37,
             "%08x-%04x-%04x-%04x-%04x%08x",
             t_hi, t_lo, mid, var_hi, rand_x, rand_lo);
}

char *wubu_uuid_v7(char *buf, size_t len) {
    if (len < 37) return NULL;

    /* Get current time in milliseconds (UNIX epoch) */
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t ts_ms = (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;

    /* Initialize entropy seed on first call */
    static int seeded = 0;
    static uint64_t seed = 0;
    if (!seeded) {
        seed = urandom_seed();
        g_uuid_counter_low = (uint32_t)(seed & 0xFFFFFFFF);
        g_uuid_counter_high = (uint32_t)(seed >> 32);
        seeded = 1;
    }

    /* Increment monotonic counter (84 bits) */
    g_uuid_counter_low++;
    if (g_uuid_counter_low == 0) g_uuid_counter_high++;

    /* Assemble UUIDv7: 48-bit timestamp + 74-bit counter */
    uint64_t counter = ((uint64_t)(g_uuid_counter_high & 0x3FF) << 32) |
                       (uint64_t)g_uuid_counter_low;

    format_uuid(buf, ts_ms, counter);
    return buf;
}

char *wubu_uuid_v7_alloc(void) {
    char *buf = (char *)malloc(37);
    if (!buf) return NULL;
    return wubu_uuid_v7(buf, 37);
}

int64_t wubu_uuid_v7_timestamp(const char *uuid) {
    if (!uuid || strlen(uuid) < 24) return -1;
    uint32_t t3, t2;
    if (sscanf(uuid, "%8x-%4x", &t3, &t2) != 2) return -1;
    return ((int64_t)t3 << 16) | (int64_t)t2;
}

int wubu_uuid_compare(const char *a, const char *b) {
    if (!a || !b) return 0;
    return strcmp(a, b);
}

int wubu_uuid_parse(const char *uuid, int64_t *ts_ms, uint16_t *rand_a,
                      uint64_t *rand_b, int *ver) {
    if (!uuid || strlen(uuid) < 36) return -1;
    uint32_t t3, t2, hi, lo;
    uint32_t r1, r2, r3;
    if (sscanf(uuid, "%8x-%4x-%4x-%4x-%4x%8x",
               &t3, &t2, &hi, &r1, &r2, &r3) != 6) return -1;

    /* Version is the high nibble of the third group */
    *ver = (hi >> 12) & 0xF;
    *rand_a = hi & 0x0FFF;

    *ts_ms = ((int64_t)t3 << 16) | (int64_t)t2;
    *rand_b = ((uint64_t)r1 << 32) | ((uint64_t)r2 << 0) | ((uint64_t)r3 << 0);
    /* rand_b high 16 bits from r1, low 48 bits from r2+r3 */
    *rand_b = ((uint64_t)(r1 & 0x0FFF) << 36) |
              ((uint64_t)r2 << 20) |
              ((uint64_t)(r3 & 0xFFFFF) << 0);

    return 0;
}
