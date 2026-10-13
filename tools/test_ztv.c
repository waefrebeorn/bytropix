/* test_ztv.c -- NAMESPACE #193 zero-trust continuous verification.
 * Proves: fresh cap allowed, revocation denies immediately, epoch bump
 * invalidates other caps (stale), unknown cap denied, telemetry, fail-closed. */
#include "wubu_ztv.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_ztv_t z;
    CHECK(wubu_ztv_init(&z) == 0, "init ok");
    CHECK(wubu_ztv_issue(&z, 100) == 0, "issue cap 100");
    CHECK(wubu_ztv_issue(&z, 200) == 0, "issue cap 200");

    /* fresh cap is allowed on every action */
    CHECK(wubu_ztv_verify(&z, 100) == 1, "cap 100 allowed (fresh)");
    CHECK(wubu_ztv_verify(&z, 100) == 1, "cap 100 allowed again (re-verified)");
    CHECK(wubu_ztv_verify(&z, 200) == 1, "cap 200 allowed (fresh)");

    /* revoke cap 100 -> immediately denied on next action */
    CHECK(wubu_ztv_revoke(&z, 100) == 0, "revoke 100");
    CHECK(wubu_ztv_verify(&z, 100) == 0, "revoked cap 100 denied immediately");

    /* the revocation bumped the epoch, so cap 200 is now STALE (even though it
     * was never revoked) -> denied. This is the zero-trust freshness guarantee. */
    CHECK(wubu_ztv_verify(&z, 200) == 0, "cap 200 denied (stale after epoch bump)");

    /* unknown cap denied */
    CHECK(wubu_ztv_verify(&z, 999) == 0, "unknown cap denied");

    /* telemetry: 3 allowed (100,100,200), 3 denied (100-revoked,200-stale,999) */
    int allowed, denied;
    wubu_ztv_stats(&z, &allowed, &denied);
    CHECK(allowed == 3, "3 allowed");
    CHECK(denied == 3, "3 denied");

    /* a cap issued AFTER the epoch bump is fresh again */
    CHECK(wubu_ztv_issue(&z, 300) == 0, "issue cap 300 (post-revocation)");
    CHECK(wubu_ztv_verify(&z, 300) == 1, "cap 300 fresh (issued at current epoch)");

    /* fail-closed */
    CHECK(wubu_ztv_issue(NULL, 1) == -1, "NULL issue -> -1");
    CHECK(wubu_ztv_revoke(NULL, 1) == -1, "NULL revoke -> -1");
    CHECK(wubu_ztv_verify(NULL, 1) == 0, "NULL verify -> 0 (fail-closed)");
    CHECK(wubu_ztv_revoke(&z, 12345) == -1, "revoke unknown -> -1");
    CHECK(wubu_ztv_init(NULL) == -1, "NULL init -> -1");

    printf("\n=== test_ztv: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
