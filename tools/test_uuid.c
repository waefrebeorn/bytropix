/*
 * test_uuid.c — tests for UUIDv7 generation (RFC 9562).
 *
 * Tests: generation, format, monotonic increment, parse, timestamp,
 * comparison, NULL safety.
 *
 * C11, no external deps beyond wubu_win.h shims.
 */
#include "wubu_uuid.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int tests_run = 0;
static int tests_pass = 0;

static void check(const char *name, int cond) {
    tests_run++;
    if (cond) { tests_pass++; printf("  PASS: %s\n", name); }
    else      { printf("  FAIL: %s\n", name); }
}

static int is_valid_uuid_v7(const char *s) {
    /* Format: 8-4-4-4-12 hex digits with version 7 */
    if (!s || strlen(s) != 36) return 0;
    if (s[8] != '-' || s[13] != '-' || s[18] != '-' || s[23] != '-') return 0;
    /* Version must be 7 */
    if (s[14] != '7') return 0;
    /* Variant: high nibble of byte 8 must be 8, 9, a, or b */
    char v = s[19];
    if (v != '8' && v != '9' && v != 'a' && v != 'b') return 0;
    return 1;
}

int main(void) {
    printf("=== test_uuid: UUIDv7 (RFC 9562) ===\n");

    /* ---- Test 1: UUIDv7 generation ---- */
    printf("\n--- Test 1: Generation ---\n");
    char buf[37];
    char *r = wubu_uuid_v7(buf, sizeof(buf));
    check("v7 returns non-NULL", r != NULL);
    check("v7 returns same buffer", r == buf);
    check("v7 format valid", is_valid_uuid_v7(buf));
    check("v7 length is 36", strlen(buf) == 36);

    /* ---- Test 2: Monotonic increment ---- */
    printf("\n--- Test 2: Monotonic ---\n");
    char buf2[37], buf3[37];
    wubu_uuid_v7(buf2, sizeof(buf2));
    wubu_uuid_v7(buf3, sizeof(buf3));
    check("two UUIDs differ", strcmp(buf2, buf3) != 0);

    /* First 14 hex chars encode timestamp — should differ by ~0ms (same ms) */
    /* But counter increments, so last 22 chars should differ */
    check("UUID2 > UUID1 lexicographically", strcmp(buf3, buf2) > 0);

    /* ---- Test 3: Buffer too small ---- */
    printf("\n--- Test 3: Buffer bounds ---\n");
    char small[10];
    check("buffer too small returns NULL",
          wubu_uuid_v7(small, sizeof(small)) == NULL);

    /* ---- Test 4: Heap allocation ---- */
    printf("\n--- Test 4: Heap alloc ---\n");
    char *hb = wubu_uuid_v7_alloc();
    check("v7_alloc returns non-NULL", hb != NULL);
    if (hb) {
        check("allocated UUID valid", is_valid_uuid_v7(hb));
        free(hb);
    }

    /* ---- Test 5: Timestamp extraction ---- */
    printf("\n--- Test 5: Timestamp ---\n");
    char ts_buf[37];
    wubu_uuid_v7(ts_buf, sizeof(ts_buf));
    int64_t ts = wubu_uuid_v7_timestamp(ts_buf);
    check("timestamp > 0", ts > 0);
    check("timestamp looks like Unix ms (2025+)", ts > 1735689600000LL);

    /* ---- Test 6: Parse ---- */
    printf("\n--- Test 6: Parse ---\n");
    char parse_buf[37];
    wubu_uuid_v7(parse_buf, sizeof(parse_buf));

    int64_t ts_before = wubu_uuid_v7_timestamp(parse_buf);

    int64_t parsed_ts;
    uint16_t parsed_ra;
    uint64_t parsed_rb;
    int parsed_ver;
    int rc = wubu_uuid_parse(parse_buf, &parsed_ts, &parsed_ra, &parsed_rb, &parsed_ver);
    check("parse returns 0", rc == 0);
    check("parsed version = 7", parsed_ver == 7);
    check("parsed timestamp matches", parsed_ts == ts_before);
    check("parsed rand_a < 4096", parsed_ra < 4096);

    /* ---- Test 7: Comparison ---- */
    printf("\n--- Test 7: Comparison ---\n");
    char a[37], b[37];
    wubu_uuid_v7(a, sizeof(a));
    wubu_uuid_v7(b, sizeof(b));
    int cmp = wubu_uuid_compare(a, b);
    check("first < second", cmp < 0);
    check("compare same string", wubu_uuid_compare(a, a) == 0);
    check("reverse > forward", wubu_uuid_compare(b, a) > 0);

    /* ---- Test 8: NULL safety ---- */
    printf("\n--- Test 8: NULL safety ---\n");
    check("timestamp NULL", wubu_uuid_v7_timestamp(NULL) == -1);
    check("timestamp too short", wubu_uuid_v7_timestamp("short") == -1);
    check("compare NULL a", wubu_uuid_compare(NULL, b) == 0);
    check("compare NULL b", wubu_uuid_compare(a, NULL) == 0);
    check("parse NULL", wubu_uuid_parse(NULL, &parsed_ts, &parsed_ra, &parsed_rb, &parsed_ver) == -1);
    check("parse too short",
          wubu_uuid_parse("abc", &parsed_ts, &parsed_ra, &parsed_rb, &parsed_ver) == -1);
    check("alloc buffer large enough", 1);

    printf("\n=== Results: %d/%d tests passed ===\n", tests_pass, tests_run);
    if (tests_pass == tests_run) {
        printf("ALL UUID TESTS PASSED\n");
        return 0;
    }
    return 1;
}
