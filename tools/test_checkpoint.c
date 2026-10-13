/* test_checkpoint.c -- BRAIN #83 checkpoint round-trips lossless.
 * Proves: save->restore is byte-for-byte identical (lossless), CRC detects
 * corruption, magic/version validated, truncation rejected, dst-too-small
 * rejected, fail-closed on NULL, empty payload works. */
#include "wubu_checkpoint.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    /* build a pseudo-random state block (deterministic) */
    unsigned char state[1024];
    uint32_t s = 0xDEADBEEF;
    for (int i = 0; i < 1024; i++) { s = s*1103515245u + 12345u; state[i] = (unsigned char)(s >> 16); }

    uint8_t ckpt[2048];
    int n = wubu_ckpt_save(state, sizeof state, ckpt, sizeof ckpt);
    CHECK(n == (int)(WUBU_CKPT_HDR + sizeof state), "save wrote header+payload");

    /* restore into a different buffer, must be byte-identical */
    unsigned char restored[1024];
    memset(restored, 0, sizeof restored);
    int r = wubu_ckpt_restore(ckpt, (size_t)n, restored, sizeof restored);
    CHECK(r == (int)sizeof state, "restore returned payload size");
    CHECK(memcmp(state, restored, sizeof state) == 0, "LOSSLESS: restored == original (byte-for-byte)");

    /* empty payload round-trips */
    uint8_t eck[64];
    int en = wubu_ckpt_save("", 0, eck, sizeof eck);
    CHECK(en == (int)WUBU_CKPT_HDR, "empty payload save = header only");
    char edst[1];
    CHECK(wubu_ckpt_restore(eck, (size_t)en, edst, 1) == 0, "empty payload restore = 0 bytes");

    /* CRC detects corruption (flip a payload byte) */
    uint8_t bad[2048];
    memcpy(bad, ckpt, (size_t)n);
    bad[WUBU_CKPT_HDR + 10] ^= 0xFF;   /* corrupt one payload byte */
    CHECK(wubu_ckpt_restore(bad, (size_t)n, restored, sizeof restored) == -1, "CRC detects corruption -> -1");

    /* bad magic rejected */
    uint8_t badmagic[2048];
    memcpy(badmagic, ckpt, (size_t)n);
    badmagic[0] ^= 0xFF;
    CHECK(wubu_ckpt_restore(badmagic, (size_t)n, restored, sizeof restored) == -1, "bad magic -> -1");

    /* bad version rejected */
    uint8_t badver[2048];
    memcpy(badver, ckpt, (size_t)n);
    badver[5] = 99;
    CHECK(wubu_ckpt_restore(badver, (size_t)n, restored, sizeof restored) == -1, "bad version -> -1");

    /* truncated buffer rejected */
    CHECK(wubu_ckpt_restore(ckpt, 8, restored, sizeof restored) == -1, "truncated header -> -1");
    CHECK(wubu_ckpt_restore(ckpt, (size_t)n - 100, restored, sizeof restored) == -1, "truncated payload -> -1");

    /* dst too small rejected */
    CHECK(wubu_ckpt_restore(ckpt, (size_t)n, restored, 10) == -1, "dst too small -> -1");

    /* save overflow rejected */
    CHECK(wubu_ckpt_save(state, sizeof state, ckpt, 100) == -1, "save overflow -> -1");

    /* fail-closed on NULL */
    CHECK(wubu_ckpt_save(NULL, 10, ckpt, sizeof ckpt) == -1, "NULL payload -> -1");
    CHECK(wubu_ckpt_restore(NULL, 10, restored, sizeof restored) == -1, "NULL buf -> -1");

    /* CRC32 known value (empty string CRC is 0) */
    CHECK(wubu_crc32("", 0) == 0, "CRC32 of empty == 0");
    CHECK(wubu_crc32("a", 1) != 0, "CRC32 of 'a' != 0");

    printf("\n=== test_checkpoint: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
