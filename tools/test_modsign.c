/* test_modsign.c -- KERNEL #23 kernel modules signed + verified on load.
 * Proves: valid signature loads, mismatched/unsigned/unknown rejected
 * (fail-closed), loaded flag, fail-closed on NULL. */
#include "wubu_modsign.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_modsign_t s;
    unsigned long long key = 0xA5A5A5A5ULL;
    CHECK(wubu_ms_init(&s, key) == 0, "init ok");
    CHECK(wubu_ms_enroll(&s, "netdrv", 0x1111ULL) == 0, "enroll netdrv");
    CHECK(wubu_ms_enroll(&s, "gfxdrv", 0x2222ULL) == 0, "enroll gfxdrv");

    /* valid signature loads */
    CHECK(wubu_ms_load(&s, "netdrv", 0x1111ULL) == 1, "valid netdrv loads");
    CHECK(wubu_ms_is_loaded(&s, "netdrv") == 1, "netdrv loaded");

    /* mismatched signature rejected */
    CHECK(wubu_ms_load(&s, "gfxdrv", 0xBADBADULL) == 0, "bad gfxdrv sig rejected");
    CHECK(wubu_ms_is_loaded(&s, "gfxdrv") == 0, "gfxdrv not loaded");

    /* correct gfxdrv sig loads */
    CHECK(wubu_ms_load(&s, "gfxdrv", 0x2222ULL) == 1, "valid gfxdrv loads");
    CHECK(wubu_ms_is_loaded(&s, "gfxdrv") == 1, "gfxdrv loaded");

    /* unknown (unenrolled) module rejected */
    CHECK(wubu_ms_load(&s, "evilmod", 0x9999ULL) == 0, "unknown module rejected");
    CHECK(wubu_ms_is_loaded(&s, "evilmod") == 0, "evilmod not loaded");

    /* zero signature (unsigned) rejected */
    CHECK(wubu_ms_load(&s, "netdrv", 0ULL) == 0, "unsigned (zero sig) rejected");

    /* fail-closed */
    CHECK(wubu_ms_init(NULL, key) == -1, "NULL init -> -1");
    CHECK(wubu_ms_enroll(NULL, "x", 1) == -1, "NULL enroll -> -1");
    CHECK(wubu_ms_load(NULL, "x", 1) == -1, "NULL load -> -1");
    CHECK(wubu_ms_is_loaded(NULL, "x") == 0, "NULL is_loaded -> 0");

    printf("\n=== test_modsign: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
