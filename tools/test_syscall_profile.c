/* test_syscall_profile.c -- OMARCHY #277 seccomp-bpf syscall filter.
 * Proves: default-deny, allow admits, dedupe, presets (none/compute/io/
 * network), fail-closed on NULL, unknown syscall denied. */
#include "wubu_syscall_profile.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_syscall_profile_t p;
    CHECK(wubu_scp_init(&p, "test") == 0, "init ok");
    CHECK(strcmp(p.name, "test") == 0, "name set");
    CHECK(wubu_scp_count(&p) == 0, "empty count 0");

    /* default-deny: nothing allowed */
    CHECK(wubu_scp_allowed(&p, 0) == 0, "read denied by default (default-deny)");
    CHECK(wubu_scp_allowed(&p, 9) == 0, "mmap denied by default");

    /* allow read + mmap */
    CHECK(wubu_scp_allow(&p, 0) == 0, "allow read");
    CHECK(wubu_scp_allow(&p, 9) == 0, "allow mmap");
    CHECK(wubu_scp_allowed(&p, 0) == 1, "read now allowed");
    CHECK(wubu_scp_allowed(&p, 9) == 1, "mmap now allowed");
    CHECK(wubu_scp_allowed(&p, 1) == 0, "write still denied");

    /* dedupe */
    CHECK(wubu_scp_allow(&p, 0) == 0, "allow read again (no-op)");
    CHECK(wubu_scp_count(&p) == 2, "count stays 2 after dedupe");

    /* negative syscall rejected */
    CHECK(wubu_scp_allow(&p, -1) == -1, "negative syscall rejected");
    CHECK(wubu_scp_allowed(&p, -1) == 0, "negative denied");

    /* preset none = default-deny */
    wubu_scp_preset(&p, "none");
    CHECK(wubu_scp_count(&p) == 0, "preset none empty");
    CHECK(wubu_scp_allowed(&p, 0) == 0, "preset none denies read");

    /* preset compute = core (read/mmap/exit allowed, socket denied) */
    wubu_scp_preset(&p, "compute");
    CHECK(wubu_scp_allowed(&p, 0) == 1, "compute allows read");
    CHECK(wubu_scp_allowed(&p, 9) == 1, "compute allows mmap");
    CHECK(wubu_scp_allowed(&p, 60) == 1, "compute allows exit");
    CHECK(wubu_scp_allowed(&p, 41) == 0, "compute denies socket");

    /* preset network = core + socket/connect */
    wubu_scp_preset(&p, "network");
    CHECK(wubu_scp_allowed(&p, 41) == 1, "network allows socket");
    CHECK(wubu_scp_allowed(&p, 42) == 1, "network allows connect");
    CHECK(wubu_scp_allowed(&p, 0) == 1, "network allows read (core)");

    /* fail-closed on NULL */
    CHECK(wubu_scp_allowed(NULL, 0) == 0, "NULL profile denies");
    CHECK(wubu_scp_allow(NULL, 0) == -1, "NULL allow rejected");
    CHECK(wubu_scp_count(NULL) == 0, "NULL count 0");

    printf("\n=== test_syscall_profile: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
