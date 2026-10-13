/* test_fw_rollback.c -- OMARCHY #288 firmware rollback on signature failure.
 * Proves: signed image verifies, mismatched signature fails (fail-closed),
 * rollback to newest known-good on failure, unknown image rejected,
 * fail-closed on NULL. */
#include "wubu_fw_rollback.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_fw_rollback_t fw;
    CHECK(wubu_fw_init(&fw) == 0, "init ok");
    /* register a known-good v1 and v2 */
    CHECK(wubu_fw_register(&fw, "fw_v1", 0xAAAAULL, 1) == 0, "register v1");
    CHECK(wubu_fw_register(&fw, "fw_v2", 0xBBBBULL, 2) == 0, "register v2");

    /* signed image verifies */
    CHECK(wubu_fw_verify(&fw, "fw_v2", 0xBBBBULL) == 1, "v2 with correct sig verifies");
    /* mismatched signature fails (fail-closed) */
    CHECK(wubu_fw_verify(&fw, "fw_v2", 0xDEADULL) == 0, "v2 with wrong sig FAILS");
    /* unknown image rejected */
    CHECK(wubu_fw_verify(&fw, "fw_evil", 0x0000ULL) == 0, "unknown image rejected");

    /* boot the correct v2 */
    int idx = wubu_fw_boot(&fw, "fw_v2", 0xBBBBULL);
    CHECK(idx == 1, "boot v2 -> index 1");

    /* a tampered v3 (wrong sig) -> rollback to newest known-good (v2) */
    int rb = wubu_fw_boot(&fw, "fw_v3", 0xBADBADULL);
    CHECK(rb == 1, "tampered image -> rollback to v2 (index 1)");

    /* boot v1 explicitly (correct sig) */
    CHECK(wubu_fw_boot(&fw, "fw_v1", 0xAAAAULL) == 0, "boot v1 -> index 0");

    /* empty firmware set -> no trustworthy image */
    wubu_fw_rollback_t empty;
    wubu_fw_init(&empty);
    CHECK(wubu_fw_boot(&empty, "x", 1) == -1, "no images -> -1 (fail-closed)");

    /* fail-closed on NULL */
    CHECK(wubu_fw_init(NULL) == -1, "NULL init -> -1");
    CHECK(wubu_fw_register(NULL, "x", 1, 1) == -1, "NULL register -> -1");
    CHECK(wubu_fw_verify(NULL, "x", 1) == 0, "NULL verify -> 0 (fail-closed)");
    CHECK(wubu_fw_boot(NULL, "x", 1) == -1, "NULL boot -> -1");

    printf("\n=== test_fw_rollback: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
