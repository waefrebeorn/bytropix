/* test_kataiso.c -- OMARCHY #290 tests: gVisor/Kata-style kernel isolation.
 *
 * Exercises the sentry boundary: classification, fail-closed default-deny,
 * sealed-policy immutability, crossing counters, and deny-rate reporting --
 * including a realistic syscall stream mixing allowed, emulated, and hostile
 * syscalls. */

#include "wubu_kataiso.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

/* illustrative x86-64 syscall numbers */
enum { SYS_READ=0, SYS_WRITE=1, SYS_OPEN=2, SYS_CLOSE=3, SYS_MMAP=9,
       SYS_EXECVE=59, SYS_KILL=62, SYS_PTRACE=101, SYS_REBOOT=169,
       SYS_INIT_MODULE=175, SYS_SOCKET=41, SYS_BIND=49 };

int main(void)
{
    wubu_kataiso_t b;

    /* --- init: default-deny --- */
    CHECK(wubu_iso_init(&b) == 0);
    CHECK(wubu_iso_lookup(&b, SYS_READ) == WUBU_ISO_DENY);   /* unclassified */

    /* --- classify a mixed policy --- */
    CHECK(wubu_iso_classify(&b, SYS_READ, WUBU_ISO_PASSTHROUGH) == 0);
    CHECK(wubu_iso_classify(&b, SYS_WRITE, WUBU_ISO_PASSTHROUGH) == 0);
    CHECK(wubu_iso_classify(&b, SYS_MMAP, WUBU_ISO_EMULATE) == 0);
    CHECK(wubu_iso_classify(&b, SYS_SOCKET, WUBU_ISO_EMULATE) == 0);
    CHECK(wubu_iso_classify(&b, SYS_PTRACE, WUBU_ISO_DENY) == 0);
    CHECK(wubu_iso_classify(&b, SYS_REBOOT, WUBU_ISO_DENY) == 0);
    CHECK(wubu_iso_classify(&b, SYS_INIT_MODULE, WUBU_ISO_DENY) == 0);

    /* --- lookup reflects policy; unclassified stays DENY --- */
    CHECK(wubu_iso_lookup(&b, SYS_READ) == WUBU_ISO_PASSTHROUGH);
    CHECK(wubu_iso_lookup(&b, SYS_MMAP) == WUBU_ISO_EMULATE);
    CHECK(wubu_iso_lookup(&b, SYS_PTRACE) == WUBU_ISO_DENY);
    CHECK(wubu_iso_lookup(&b, SYS_EXECVE) == WUBU_ISO_DENY);   /* not classified */

    /* --- override a rule (later wins) --- */
    CHECK(wubu_iso_classify(&b, SYS_READ, WUBU_ISO_EMULATE) == 0);
    CHECK(wubu_iso_lookup(&b, SYS_READ) == WUBU_ISO_EMULATE);
    CHECK(wubu_iso_classify(&b, SYS_READ, WUBU_ISO_PASSTHROUGH) == 0);  /* restore */

    /* --- seal: no more rules --- */
    CHECK(wubu_iso_seal(&b) == 0);
    CHECK(wubu_iso_seal(&b) == -1);               /* already sealed */
    CHECK(wubu_iso_classify(&b, SYS_KILL, WUBU_ISO_PASSTHROUGH) == -1);
    CHECK(wubu_iso_lookup(&b, SYS_KILL) == WUBU_ISO_DENY);   /* still denied */

    /* --- crossing a realistic stream --- */
    int stream[] = { SYS_READ, SYS_WRITE, SYS_MMAP, SYS_EXECVE, SYS_PTRACE,
                     SYS_OPEN, SYS_REBOOT, SYS_SOCKET, SYS_KILL, SYS_CLOSE };
    int n = (int)(sizeof stream / sizeof stream[0]);
    for (int i = 0; i < n; i++) wubu_iso_cross(&b, stream[i]);
    /* READ=pass, WRITE=pass, MMAP=emulate, EXECVE=deny, PTRACE=deny,
       OPEN=deny(unclassified), REBOOT=deny, SOCKET=emulate, KILL=deny,
       CLOSE=deny(unclassified) */
    CHECK(b.n_total == (uint64_t)n);
    CHECK(b.n_passthrough == 2);   /* read, write */
    CHECK(b.n_emulate == 2);       /* mmap, socket */
    CHECK(b.n_deny == 6);          /* execve, ptrace, open, reboot, kill, close */

    /* --- deny rate: 6/10 = 600 per-mille --- */
    CHECK(wubu_iso_deny_per_mille(&b) == 600);

    /* --- NULL boundary is fail-closed --- */
    CHECK(wubu_iso_cross(NULL, SYS_READ) == WUBU_ISO_DENY);
    CHECK(wubu_iso_lookup(NULL, SYS_READ) == WUBU_ISO_DENY);

    /* --- action names --- */
    CHECK(strcmp(wubu_iso_action_name(WUBU_ISO_PASSTHROUGH), "passthrough") == 0);
    CHECK(strcmp(wubu_iso_action_name(WUBU_ISO_EMULATE), "emulate") == 0);
    CHECK(strcmp(wubu_iso_action_name(WUBU_ISO_DENY), "deny") == 0);

    /* --- empty boundary: 0 crossings -> 0 per-mille --- */
    {
        wubu_kataiso_t e;
        CHECK(wubu_iso_init(&e) == 0);
        CHECK(wubu_iso_deny_per_mille(&e) == 0);
    }

    if (fails == 0) printf("ALL KATAISO TESTS PASSED\n");
    else printf("%d KATAISO TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
