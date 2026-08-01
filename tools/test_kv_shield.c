/*
 * test_kv_shield.c -- L15 KVShield adversarial-robustness verification.
 */
#include "wubu_kv_shield.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static int failures = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL: %s\n", msg); failures++; } } while (0)

/* trivial identity remap */
static long ident(void *ud, long logical) { (void)ud; return logical; }

int main(void) {
    printf("=== test_kv_shield (L15) ===\n");

    /* bounds check */
    CHECK(wubu_kv_shield_check(0, 10, NULL) == 1, "idx 0 ok");
    CHECK(wubu_kv_shield_check(9, 10, NULL) == 1, "idx cap-1 ok");
    CHECK(wubu_kv_shield_check(10, 10, NULL) == 0, "idx == cap rejected");
    CHECK(wubu_kv_shield_check(-1, 10, NULL) == 0, "negative rejected");
    CHECK(wubu_kv_shield_check(0, 0, NULL) == 0, "cap 0 rejects all");

    /* remap path */
    wubu_kv_shield_remap r = { ident, NULL };
    CHECK(wubu_kv_shield_check(5, 10, &r) == 1, "remap idx 5 ok");
    CHECK(wubu_kv_shield_check(99, 10, &r) == 0, "remap idx 99 rejected");

    /* safe read: buffer of 10 slots x 4 bytes. */
    unsigned char buf[40];
    for (int i = 0; i < 40; i++) buf[i] = (unsigned char)(i + 1);
    unsigned char out[4] = {0};
    int n = wubu_kv_shield_read(buf, 2, 10, 4, out, 4); /* slot 2 -> bytes 9..12 */
    CHECK(n == 4, "read 4 bytes");
    CHECK(out[0] == 9 && out[3] == 12, "read correct slot bytes (9..12)");
    /* OOB read rejected */
    CHECK(wubu_kv_shield_read(buf, 10, 10, 4, out, 4) == 0, "OOB read rejected");
    CHECK(wubu_kv_shield_read(buf, -1, 10, 4, out, 4) == 0, "negative read rejected");
    /* n clamped to slot_bytes */
    unsigned char out2[8] = {0};
    int n2 = wubu_kv_shield_read(buf, 0, 10, 4, out2, 8);
    CHECK(n2 == 4, "n clamped to slot_bytes");

    if (failures == 0) { printf("ALL KV-SHIELD TESTS PASSED\n"); return 0; }
    printf("%d KV-SHIELD TEST(S) FAILED\n", failures);
    return 1;
}
