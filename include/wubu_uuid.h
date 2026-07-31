#ifndef WUBU_UUID_H
#define WUBU_UUID_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * wubu_uuid.h — UUIDv7 generation for WuBuOS/WuBuWizard session tracking.
 *
 * Why UUIDv7 (RFC 9562): monotonic timestamp-based UUIDs that are
 * lexicographically sortable — critical for KV cache session ordering,
 * EDR audit trails, and GDPR consent logs. No UUIDv4 randomness needed
 * (deterministic, reproducible).
 *
 * Privacy: UUIDv7 contains ONLY a timestamp + counter. No machine hardware
 * info, no MAC address, no user identity. Fully GDPR-compliant as a session
 * identifier — it cannot be reverse-linked to a person.
 *
 * Ohio single-developer compliance: these UUIDs are used solely for
 * internal session tracking and audit logging. No external transmission.
 */

/* Generate a UUIDv7 string (RFC 9562 format: xxxxxxxx-xxxx-7xxx-xxxx-xxxxxxxxxxxx).
 * Fills caller-provided buffer (must be >= 37 bytes). Returns the same buffer. */
char *wubu_uuid_v7(char *buf, size_t len);

/* Same but returns a heap-allocated string (caller must free). */
char *wubu_uuid_v7_alloc(void);

/* Extract the Unix timestamp (seconds) from a UUIDv7 string. Returns -1 on parse error. */
int64_t wubu_uuid_v7_timestamp(const char *uuid);

/* Compare two UUIDv7 strings lexicographically. Returns -1, 0, +1. */
int wubu_uuid_compare(const char *a, const char *b);

/* Parse a UUIDv7 into its components: ts_ms (unix millis), rand_a (12 bits),
 * rand_b (62 bits), ver (should be 7). Returns 0 on success. */
int wubu_uuid_parse(const char *uuid, int64_t *ts_ms, uint16_t *rand_a,
                      uint64_t *rand_b, int *ver);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_UUID_H */
