/* test_nnp.c -- OMARCHY #284 NoNewPrivileges + PrivateTmp default.
 * Proves: enforce forces both on, audit passes only when both on, disable
 * requests always refused, fail-closed on NULL. */
#include "wubu_nnp.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    /* a sandbox with both off (non-compliant) */
    wubu_unit_sandbox_t s = {0};
    s.no_new_privileges = 0;
    s.private_tmp = 0;
    CHECK(wubu_nnp_audit(&s) == 0, "both off -> non-compliant");

    /* enforce forces both on */
    CHECK(wubu_nnp_enforce(&s) == 0, "enforce ok");
    CHECK(s.no_new_privileges == 1, "NoNewPrivileges forced on");
    CHECK(s.private_tmp == 1, "PrivateTmp forced on");
    CHECK(wubu_nnp_audit(&s) == 1, "compliant after enforce");

    /* disable requests always refused (non-negotiable) */
    CHECK(wubu_nnp_may_disable() == 0, "cannot disable NoNewPrivileges");
    CHECK(wubu_nnp_may_disable_tmp() == 0, "cannot disable PrivateTmp");

    /* audit fails if either is turned off */
    s.no_new_privileges = 0;
    CHECK(wubu_nnp_audit(&s) == 0, "NoNewPrivileges off -> non-compliant");
    s.no_new_privileges = 1; s.private_tmp = 0;
    CHECK(wubu_nnp_audit(&s) == 0, "PrivateTmp off -> non-compliant");

    /* fail-closed on NULL */
    CHECK(wubu_nnp_audit(NULL) == 0, "NULL audit -> 0 (fail-closed)");
    CHECK(wubu_nnp_enforce(NULL) == -1, "NULL enforce -> -1");

    printf("\n=== test_nnp: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
