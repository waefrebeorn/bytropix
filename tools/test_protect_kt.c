/* test_protect_kt.c -- OMARCHY #285 ProtectHome + ProtectKernelTunables.
 * Proves: kernel tunables read-only when protected, home read-only/inaccessible
 * per policy, other paths allowed, fail-closed on NULL/unset. */
#include "wubu_protect_kt.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_protect_kt_t k;
    CHECK(wubu_kt_init(&k) == 0, "init ok (hardened)");

    /* kernel tunables read-only */
    CHECK(wubu_kt_is_kernel_path("/proc/sys/kernel/hostname") == 1, "/proc/sys is kernel path");
    CHECK(wubu_kt_is_kernel_path("/sys/kernel") == 1, "/sys is kernel path");
    CHECK(wubu_kt_is_kernel_path("/proc/sysrq-trigger") == 1, "sysrq-trigger is kernel path");
    CHECK(wubu_kt_is_kernel_path("/home/x") == 0, "/home not kernel path");
    CHECK(wubu_kt_access(&k, "/proc/sys/kernel/hostname") == WUBU_KT_READONLY, "kernel tunable read-only");
    CHECK(wubu_kt_access(&k, "/sys/kernel/x") == WUBU_KT_READONLY, "/sys read-only");

    /* home read-only (default policy=1) */
    CHECK(wubu_kt_access(&k, "/home/wubu/x") == WUBU_KT_READONLY, "home read-only");
    CHECK(wubu_kt_access(&k, "/root/x") == WUBU_KT_READONLY, "root read-only");

    /* home inaccessible (policy=2) */
    k.protect_home = 2;
    CHECK(wubu_kt_access(&k, "/home/wubu/x") == WUBU_KT_DENY, "home inaccessible");

    /* home allowed (policy=0) */
    k.protect_home = 0;
    CHECK(wubu_kt_access(&k, "/home/wubu/x") == WUBU_KT_ALLOW, "home allowed when unprotected");

    /* kernel tunables unprotected -> deny (fail-closed) */
    k.protect_kernel_tunables = 0;
    CHECK(wubu_kt_access(&k, "/proc/sys/x") == WUBU_KT_DENY, "kernel tunable unprotected -> deny (fail-closed)");

    /* other paths allowed */
    CHECK(wubu_kt_access(&k, "/opt/app/data") == WUBU_KT_ALLOW, "other path allowed");

    /* fail-closed */
    CHECK(wubu_kt_access(NULL, "/x") == WUBU_KT_DENY, "NULL policy -> deny");
    CHECK(wubu_kt_access(&k, NULL) == WUBU_KT_DENY, "NULL path -> deny");
    CHECK(wubu_kt_is_kernel_path(NULL) == 0, "NULL kernel path -> 0");

    printf("\n=== test_protect_kt: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
