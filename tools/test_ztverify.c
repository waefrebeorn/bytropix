/* test_ztverify.c -- AGIEDR #301 zero-trust continuous re-verification.
 * Proves: enrolled agent trusted, within-tolerance re-verify keeps trust +
 * updates expected, drift beyond tolerance revokes (fail-closed), unknown
 * agent denied, operator revoke, stats, fail-closed on NULL. */
#include "wubu_ztverify.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_ztverify_t z;
    CHECK(wubu_ztverify_init(&z) == 0, "init ok");
    CHECK(wubu_ztverify_enroll(&z, 42, 100.0, 5.0) == 0, "enroll agent 42 (att=100 tol=5)");
    CHECK(wubu_ztverify_trusted(&z, 42) == 1, "enrolled agent trusted");

    /* within tolerance -> stays trusted, expected updates (continuous) */
    CHECK(wubu_ztverify_check(&z, 42, 102.0) == 1, "att 102 within tol 5 -> trusted");
    CHECK(wubu_ztverify_check(&z, 42, 103.0) == 1, "att 103 (drift 1 from new expected) -> trusted");

    /* drift beyond tolerance -> revoked (fail-closed) */
    CHECK(wubu_ztverify_check(&z, 42, 200.0) == 0, "att 200 (drift 97) -> revoked");
    CHECK(wubu_ztverify_trusted(&z, 42) == 0, "agent 42 now untrusted");

    /* once revoked, stays revoked: re-attesting at the drifted value does
     * NOT silently restore trust (zero-trust: an attacker who drifted it
     * could just re-attest). Only re-enrollment by an operator restores. */
    CHECK(wubu_ztverify_check(&z, 42, 200.0) == 0, "revoked agent stays revoked (no silent recovery)");
    CHECK(wubu_ztverify_trusted(&z, 42) == 0, "agent 42 still untrusted");

    /* unknown agent -> fail-closed deny */
    CHECK(wubu_ztverify_check(&z, 999, 1.0) == 0, "unknown agent check -> 0 (deny)");
    CHECK(wubu_ztverify_trusted(&z, 999) == 0, "unknown agent not trusted");

    /* operator revoke */
    wubu_ztverify_enroll(&z, 7, 50.0, 1.0);
    CHECK(wubu_ztverify_trusted(&z, 7) == 1, "agent 7 trusted");
    CHECK(wubu_ztverify_revoke(&z, 7) == 0, "operator revoke ok");
    CHECK(wubu_ztverify_trusted(&z, 7) == 0, "agent 7 revoked");
    CHECK(wubu_ztverify_revokes(&z, 7) == 1, "revoke count 1");

    /* drift count for agent 42 (2 drift revokes: initial 200 + re-check 200) */
    CHECK(wubu_ztverify_revokes(&z, 42) == 2, "agent 42 had 2 drift revokes");

    /* fail-closed on NULL */
    CHECK(wubu_ztverify_check(NULL, 42, 1.0) == 0, "NULL check -> 0");
    CHECK(wubu_ztverify_trusted(NULL, 42) == 0, "NULL trusted -> 0");
    CHECK(wubu_ztverify_enroll(NULL, 1, 1, 1) == -1, "NULL enroll -> -1");
    CHECK(wubu_ztverify_revoke(NULL, 42) == -1, "NULL revoke -> -1");

    printf("\n=== test_ztverify: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
