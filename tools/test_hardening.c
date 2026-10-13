/* test_hardening.c -- OMARCHY #283 systemd hardening profile catalog.
 * Proves: each profile applies the expected directives, score increases with
 * strictness (baseline < balanced < strict < paranoid), paranoid is most
 * confined, fail-closed on NULL/bad profile. */
#include "wubu_hardening.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_unit_sandbox_t s;

    /* baseline: only ProtectSystem=strict */
    CHECK(wubu_hardening_apply(&s, WUBU_HARD_BASELINE) == 0, "apply baseline");
    CHECK(s.protect_system == WUBU_PS_STRICT, "baseline: ProtectSystem=strict");
    CHECK(s.protect_home == WUBU_PH_NO, "baseline: ProtectHome=no");
    CHECK(s.private_tmp == 0, "baseline: no PrivateTmp");
    CHECK(s.no_new_privileges == 0, "baseline: no NoNewPrivileges");
    int base_score = wubu_hardening_score(&s);

    /* balanced: + ProtectHome RO, PrivateTmp, NoNewPrivileges */
    wubu_hardening_apply(&s, WUBU_HARD_BALANCED);
    CHECK(s.protect_home == WUBU_PH_READONLY, "balanced: ProtectHome=read-only");
    CHECK(s.private_tmp == 1, "balanced: PrivateTmp");
    CHECK(s.no_new_privileges == 1, "balanced: NoNewPrivileges");
    CHECK(s.restrict_namespaces == 0, "balanced: no RestrictNamespaces");
    int bal_score = wubu_hardening_score(&s);

    /* strict: + RestrictNamespaces */
    wubu_hardening_apply(&s, WUBU_HARD_STRICT);
    CHECK(s.restrict_namespaces == 1, "strict: RestrictNamespaces");
    int strict_score = wubu_hardening_score(&s);

    /* paranoid: ProtectHome=yes (fully inaccessible) */
    wubu_hardening_apply(&s, WUBU_HARD_PARANOID);
    CHECK(s.protect_home == WUBU_PH_YES, "paranoid: ProtectHome=yes");
    CHECK(wubu_unit_sandbox_fs_access(&s, "/home/wubu/x") == WUBU_FS_DENY, "paranoid: /home denied");
    int par_score = wubu_hardening_score(&s);

    /* score monotonically increases with strictness */
    CHECK(base_score < bal_score, "baseline < balanced score");
    CHECK(bal_score < strict_score, "balanced < strict score");
    CHECK(strict_score < par_score, "strict < paranoid score");
    CHECK(par_score <= 100, "score capped at 100");

    /* profile names */
    CHECK(strcmp(wubu_hardening_name(WUBU_HARD_BASELINE), "baseline") == 0, "name baseline");
    CHECK(strcmp(wubu_hardening_name(WUBU_HARD_PARANOID), "paranoid") == 0, "name paranoid");

    /* fail-closed */
    CHECK(wubu_hardening_apply(NULL, WUBU_HARD_STRICT) == -1, "NULL apply -> -1");
    CHECK(wubu_hardening_apply(&s, (wubu_hard_profile_t)99) == -1, "bad profile -> -1");
    CHECK(wubu_hardening_score(NULL) == 0, "NULL score -> 0");

    printf("\n=== test_hardening: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
