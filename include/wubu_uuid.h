/*
 * wubu_uuid.h — UUIDv7 generation (RFC 9562) for session tracking.
 *
 * Ported from WuBuOS/src/runtime/wubu_uuid.c to wubuwizard Windows build.
 * Pure C11, self-contained, uses wubu_win.h for POSIX shims on Windows.
 *
 * Why UUIDv7: monotonic timestamp-based UUIDs, lexicographically sortable.
 * Privacy: contains ONLY timestamp + counter. No MAC/hardware info.
 *
 * SPDX-License-Identifier: Waefrebeorn-UMV3
 */
#ifndef WUBU_UUID_H
#define WUBU_UUID_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Generate a UUIDv7 string (RFC 9562 format: xxxxxxxx-xxxx-7xxx-xxxx-xxxxxxxxxxxx).
 * Fills caller-provided buffer (must be >= 37 bytes). Returns the same buffer. */
char *wubu_uuid_v7(char *buf, size_t len);

/* Heap-allocated UUIDv7 string (caller must free). */
char *wubu_uuid_v7_alloc(void);

/* Extract Unix timestamp (seconds) from UUIDv7. Returns -1 on parse error. */
int64_t wubu_uuid_v7_timestamp(const char *uuid);

/* Compare two UUIDv7 strings lexicographically. Returns -1, 0, +1. */
int wubu_uuid_compare(const char *a, const char *b);

/* Parse UUIDv7 into components: ts_ms (unix millis), rand_a (12 bits),
 * rand_b (62 bits), ver (should be 7). Returns 0 on success. */
int wubu_uuid_parse(const char *uuid, int64_t *ts_ms, uint16_t *rand_a,
                    uint64_t *rand_b, int *ver);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_UUID_H */
