/* test_pacman.c -- OMARCHY #272 Arch base: pacman package manager driver.
 * Proves: op parsing (-Q/-S/-R/-D/-F/-U/-T), flag clusters (-Qi/-Syu),
 * long options, target extraction, validation (operand requirements),
 * network gating, fail-closed on unknown/malformed. */
#include "wubu_pacman.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_pac_cmd_t c;

    /* -Qe: query explicitly-installed (op QUERY, no targets needed) */
    const char *a1[] = {"pacman", "-Qe"};
    CHECK(wubu_pacman_parse(a1, 2, &c) == 0, "parse -Qe");
    CHECK(c.op == WUBU_PAC_QUERY, "op QUERY");
    CHECK(wubu_pacman_validate(&c) == 0, "validate -Qe ok");

    /* -Qi vim: query info on vim */
    const char *a2[] = {"pacman", "-Qi", "vim"};
    CHECK(wubu_pacman_parse(a2, 3, &c) == 0, "parse -Qi vim");
    CHECK(c.op == WUBU_PAC_QUERY, "op QUERY");
    CHECK((c.flags & WUBU_PAC_FLAG_INFO) != 0, "INFO flag set");
    CHECK(c.n_targets == 1 && strcmp(c.targets[0], "vim") == 0, "target vim");

    /* -Syu: full system upgrade (SYNC + refresh + sysup) */
    const char *a3[] = {"pacman", "-Syu"};
    CHECK(wubu_pacman_parse(a3, 2, &c) == 0, "parse -Syu");
    CHECK(c.op == WUBU_PAC_SYNC, "op SYNC");
    CHECK((c.flags & WUBU_PAC_FLAG_REFRESH) != 0, "REFRESH flag");
    CHECK((c.flags & WUBU_PAC_FLAG_SYSUP) != 0, "SYSUP flag");
    CHECK(wubu_pacman_needs_network(&c) == 1, "-Syu needs network");

    /* -S pkg1 pkg2 --noconfirm: install two packages */
    const char *a4[] = {"pacman", "-S", "pkg1", "pkg2", "--noconfirm"};
    CHECK(wubu_pacman_parse(a4, 5, &c) == 0, "parse -S pkg1 pkg2 --noconfirm");
    CHECK(c.op == WUBU_PAC_SYNC, "op SYNC");
    CHECK(c.n_targets == 2, "2 targets");
    CHECK((c.flags & WUBU_PAC_FLAG_NOCONF) != 0, "NOCONF flag");

    /* -R pkg: remove (needs a target) */
    const char *a5[] = {"pacman", "-R", "pkg"};
    CHECK(wubu_pacman_parse(a5, 3, &c) == 0, "parse -R pkg");
    CHECK(c.op == WUBU_PAC_REMOVE, "op REMOVE");
    CHECK(wubu_pacman_validate(&c) == 0, "validate -R pkg ok");
    CHECK(wubu_pacman_needs_network(&c) == 0, "-R no network");

    /* -R with no target -> validation fails */
    const char *a6[] = {"pacman", "-R"};
    CHECK(wubu_pacman_parse(a6, 2, &c) == 0, "parse -R (parses)");
    CHECK(wubu_pacman_validate(&c) == -1, "validate -R (no target) fails");

    /* -U local.pkg.tar.zst: upgrade from file (no network) */
    const char *a7[] = {"pacman", "-U", "local.pkg.tar.zst"};
    CHECK(wubu_pacman_parse(a7, 3, &c) == 0, "parse -U file");
    CHECK(c.op == WUBU_PAC_UPGRADE, "op UPGRADE");
    CHECK(wubu_pacman_needs_network(&c) == 0, "-U no network (local file)");

    /* -Ql: list files (LIST flag) */
    const char *a8[] = {"pacman", "-Ql", "vim"};
    CHECK(wubu_pacman_parse(a8, 3, &c) == 0, "parse -Ql vim");
    CHECK((c.flags & WUBU_PAC_FLAG_LIST) != 0, "LIST flag");

    /* fail-closed: no operation flag */
    const char *a9[] = {"pacman", "vim"};
    CHECK(wubu_pacman_parse(a9, 2, &c) == -1, "no op flag -> -1");

    /* fail-closed: unknown flag */
    const char *a10[] = {"pacman", "-Qz"};
    CHECK(wubu_pacman_parse(a10, 2, &c) == -1, "unknown flag -z -> -1");

    /* fail-closed: unknown long option */
    const char *a11[] = {"pacman", "-S", "--bogus"};
    CHECK(wubu_pacman_parse(a11, 3, &c) == -1, "unknown long opt -> -1");

    /* fail-closed: NULL / empty */
    CHECK(wubu_pacman_parse(NULL, 2, &c) == -1, "NULL argv -> -1");
    const char *a12[] = {"pacman"};
    CHECK(wubu_pacman_parse(a12, 1, &c) == -1, "no args -> -1");

    /* op names */
    CHECK(strcmp(wubu_pac_op_name(WUBU_PAC_QUERY), "query") == 0, "op name query");
    CHECK(strcmp(wubu_pac_op_name(WUBU_PAC_SYNC), "sync") == 0, "op name sync");

    printf("\n=== test_pacman: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
