/* test_unit_sandbox.c -- OMARCHY #275 systemd unit sandboxing (ProtectSystem).
 * Proves: hardened defaults, directive parsing, fs access verdicts for
 * /usr//etc (RO under strict), /home (RO under read-only, deny under yes),
 * /tmp (private allow), escalation gating, fail-closed on NULL. */
#include "wubu_unit_sandbox.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_unit_sandbox_t s;
    CHECK(wubu_unit_sandbox_init_hardened(&s) == 0, "hardened init ok");
    CHECK(s.protect_system == WUBU_PS_STRICT, "default ProtectSystem=strict");
    CHECK(s.protect_home == WUBU_PH_READONLY, "default ProtectHome=read-only");
    CHECK(s.private_tmp == 1, "default PrivateTmp");
    CHECK(s.no_new_privileges == 1, "default NoNewPrivileges");
    CHECK(s.restrict_namespaces == 1, "default RestrictNamespaces");

    /* directive parsing (case-insensitive) */
    CHECK(wubu_unit_sandbox_parse(&s, "protectsystem", "full") == 0, "parse ProtectSystem=full (ci)");
    CHECK(s.protect_system == WUBU_PS_FULL, "applied full");
    CHECK(wubu_unit_sandbox_parse(&s, "ProtectHome", "yes") == 0, "parse ProtectHome=yes");
    CHECK(s.protect_home == WUBU_PH_YES, "applied yes");
    CHECK(wubu_unit_sandbox_parse(&s, "PrivateTmp", "no") == 0, "parse PrivateTmp=no");
    CHECK(s.private_tmp == 0, "applied no");
    CHECK(wubu_unit_sandbox_parse(&s, "Bogus", "x") == -1, "unknown directive -> -1");

    /* fs access: strict makes /usr and /etc read-only */
    wubu_unit_sandbox_t st;
    wubu_unit_sandbox_init_hardened(&st);   /* strict */
    CHECK(wubu_unit_sandbox_fs_access(&st, "/usr/bin/foo") == WUBU_FS_READONLY, "/usr RO under strict");
    CHECK(wubu_unit_sandbox_fs_access(&st, "/etc/passwd") == WUBU_FS_READONLY, "/etc RO under strict");
    CHECK(wubu_unit_sandbox_fs_access(&st, "/boot/vmlinuz") == WUBU_FS_READONLY, "/boot RO under strict");
    /* arbitrary path also RO under strict (whole hierarchy read-only) */
    CHECK(wubu_unit_sandbox_fs_access(&st, "/srv/app/data") == WUBU_FS_READONLY, "other path RO under strict");

    /* /home read-only under ProtectHome=read-only */
    CHECK(wubu_unit_sandbox_fs_access(&st, "/home/wubu/x") == WUBU_FS_READONLY, "/home RO under read-only");
    /* /home deny under ProtectHome=yes */
    wubu_unit_sandbox_parse(&st, "ProtectHome", "yes");
    CHECK(wubu_unit_sandbox_fs_access(&st, "/home/wubu/x") == WUBU_FS_DENY, "/home DENY under yes");

    /* /tmp private allow (isolated) */
    CHECK(wubu_unit_sandbox_fs_access(&st, "/tmp/scratch") == WUBU_FS_ALLOW, "/tmp private allow");

    /* ProtectSystem=no -> writable */
    wubu_unit_sandbox_parse(&st, "ProtectSystem", "no");
    CHECK(wubu_unit_sandbox_fs_access(&st, "/usr/bin/foo") == WUBU_FS_ALLOW, "/usr allow under no");

    /* escalation gating */
    wubu_unit_sandbox_t esc;
    wubu_unit_sandbox_init_hardened(&esc);
    CHECK(wubu_unit_sandbox_may_escalate(&esc) == 0, "hardened cannot escalate");
    wubu_unit_sandbox_parse(&esc, "NoNewPrivileges", "no");
    wubu_unit_sandbox_parse(&esc, "RestrictNamespaces", "no");
    CHECK(wubu_unit_sandbox_may_escalate(&esc) == 1, "both off -> may escalate");

    /* fail-closed on NULL */
    CHECK(wubu_unit_sandbox_fs_access(NULL, "/x") == WUBU_FS_DENY, "NULL profile -> DENY");
    CHECK(wubu_unit_sandbox_fs_access(&st, NULL) == WUBU_FS_DENY, "NULL path -> DENY");
    CHECK(wubu_unit_sandbox_may_escalate(NULL) == 0, "NULL -> no escalate");

    printf("\n=== test_unit_sandbox: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
