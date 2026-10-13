/* test_ct_sandbox.c -- RUNTIME #341 tests: container isolation cgroups+seccomp. */
#include "wubu_ct_sandbox.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

enum { SC_READ=0, SC_WRITE=1, SC_OPEN=2, SC_CLOSE=3, SC_MMAP=9, SC_EXECVE=59, SC_PTRACE=101, SC_REBOOT=169 };

int main(void)
{
    wubu_sandbox_t s;

    CHECK(wubu_sb_init(&s) == 0);
    /* create a container: 50% cpu, 1MB mem, 64 pids, 10MB/s io */
    CHECK(wubu_sb_create(&s, "app", 50000, 100000, 1048576, 64, 10485760) == 0);
    CHECK(wubu_sb_create(&s, "app", 0,0,0,0,0) == -1);   /* dup */
    CHECK(wubu_sb_find(&s, "app") == 0);
    CHECK(wubu_sb_find(&s, "nope") == -1);

    /* seccomp: default-deny, allow a safe set */
    CHECK(wubu_sb_allow_syscall(&s, "app", SC_READ) == 0);
    CHECK(wubu_sb_allow_syscall(&s, "app", SC_WRITE) == 0);
    CHECK(wubu_sb_allow_syscall(&s, "app", SC_MMAP) == 0);
    CHECK(wubu_sb_allow_syscall(&s, "app", SC_READ) == 0);   /* dup allow ok */

    /* allowed syscalls admitted */
    CHECK(wubu_sb_syscall(&s, "app", SC_READ) == 1);
    CHECK(wubu_sb_syscall(&s, "app", SC_WRITE) == 1);
    /* dangerous syscalls blocked (default-deny) */
    CHECK(wubu_sb_syscall(&s, "app", SC_PTRACE) == 0);
    CHECK(wubu_sb_syscall(&s, "app", SC_REBOOT) == 0);
    CHECK(wubu_sb_syscall(&s, "app", SC_EXECVE) == 0);
    CHECK(s.cts[0].n_admitted == 2);
    CHECK(s.cts[0].n_blocked == 3);

    /* switch to default-allow: unlisted now admitted */
    CHECK(wubu_sb_set_default_deny(&s, "app", 0) == 0);
    CHECK(wubu_sb_syscall(&s, "app", SC_PTRACE) == 1);
    CHECK(wubu_sb_isolated(&s, "app") == 0);   /* no filter = not isolated */
    CHECK(wubu_sb_set_default_deny(&s, "app", 1) == 0);   /* restore */

    /* resource accounting + limits */
    CHECK(wubu_sb_isolated(&s, "app") == 1);   /* within limits, filtered */
    CHECK(wubu_sb_account(&s, "app", 40000, 500000, 30, 5000000) == 0);
    CHECK(wubu_sb_isolated(&s, "app") == 1);
    CHECK(wubu_sb_cpu_throttled(&s, "app") == 0);
    CHECK(wubu_sb_account(&s, "app", 10000, 0, 0, 0) == 0);   /* cpu used = 50000 = quota */
    CHECK(wubu_sb_cpu_throttled(&s, "app") == 1);
    CHECK(wubu_sb_isolated(&s, "app") == 0);   /* cpu throttled -> not isolated */

    /* mem exceeded */
    CHECK(wubu_sb_account(&s, "app", 0, 600000, 0, 0) == 0);  /* mem = 1.1MB > 1MB */
    CHECK(wubu_sb_mem_exceeded(&s, "app") == 1);
    /* pids exceeded */
    CHECK(wubu_sb_account(&s, "app", 0, 0, 40, 0) == 0);      /* pids = 70 > 64 */
    CHECK(wubu_sb_pids_exceeded(&s, "app") == 1);
    /* io exceeded */
    CHECK(wubu_sb_account(&s, "app", 0, 0, 0, 6000000) == 0); /* io = 11MB > 10MB */
    CHECK(wubu_sb_io_exceeded(&s, "app") == 1);

    /* unlimited container: never exceeds */
    CHECK(wubu_sb_create(&s, "free", 0, 0, 0, 0, 0) == 0);
    CHECK(wubu_sb_allow_syscall(&s, "free", SC_READ) == 0);
    CHECK(wubu_sb_account(&s, "free", 999999, 999999, 999, 999999) == 0);
    CHECK(wubu_sb_cpu_throttled(&s, "free") == 0);
    CHECK(wubu_sb_mem_exceeded(&s, "free") == 0);
    CHECK(wubu_sb_pids_exceeded(&s, "free") == 0);
    CHECK(wubu_sb_io_exceeded(&s, "free") == 0);
    CHECK(wubu_sb_isolated(&s, "free") == 1);   /* filtered + unlimited */

    /* bad args */
    CHECK(wubu_sb_syscall(&s, "nope", SC_READ) == -1);
    CHECK(wubu_sb_allow_syscall(&s, "nope", SC_READ) == -1);
    CHECK(wubu_sb_account(&s, "nope", 1,1,1,1) == -1);
    CHECK(wubu_sb_isolated(&s, "nope") == -1);
    CHECK(wubu_sb_create(&s, "bad", -1, 100000, 0, 0, 0) == -1);  /* negative quota */

    if (fails == 0) printf("ALL CT_SANDBOX TESTS PASSED\n");
    else printf("%d CT_SANDBOX TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
