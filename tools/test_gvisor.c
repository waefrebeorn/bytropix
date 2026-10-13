/* test_gvisor.c -- RUNTIME #351 tests: gVisor user-kernel isolation. */
#include "wubu_gvisor.h"
#include <stdio.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

enum { SYS_GETPID=39, SYS_READ=0, SYS_WRITE=1, SYS_OPEN=2, SYS_CLOSE=3, SYS_FSTAT=5,
       SYS_BRK=12, SYS_MMAP=9, SYS_EXIT=60, SYS_GETTIMEOFDAY=78, SYS_EPOLL_WAIT=256 };

int main(void)
{
    wubu_gvisor_t gv;

    /* init: deny-by-default, seccomp off, sentry down */
    CHECK(wubu_gv_init(&gv) == 0);
    CHECK(wubu_gv_n_trapped(&gv) == 0);
    CHECK(gv.seccomp_on == 0);
    CHECK(gv.sentry_ready == 0);

    /* register a small syscall set (deny-by-default until admitted) */
    CHECK(wubu_gv_register_syscall(&gv, SYS_READ, "read") == 0);
    CHECK(wubu_gv_register_syscall(&gv, SYS_WRITE, "write") == 0);
    CHECK(wubu_gv_register_syscall(&gv, SYS_OPEN, "open") == 0);
    CHECK(wubu_gv_register_syscall(&gv, SYS_CLOSE, "close") == 0);
    CHECK(wubu_gv_register_syscall(&gv, SYS_GETPID, "getpid") == 0);
    CHECK(gv.n == 5);
    /* duplicate registration rejected */
    CHECK(wubu_gv_register_syscall(&gv, SYS_READ, "read") == -1);
    /* bad args */
    CHECK(wubu_gv_register_syscall(&gv, -1, "x") == -1);
    CHECK(wubu_gv_register_syscall(&gv, SYS_READ, NULL) == -1);

    /* admit only read+getpid; write/open/close remain denied */
    CHECK(wubu_gv_admit(&gv, SYS_READ) == 0);
    CHECK(wubu_gv_admit(&gv, SYS_GETPID) == 0);
    CHECK(wubu_gv_admit(&gv, SYS_EXIT) == -1);   /* not registered */
    CHECK(gv.n_admitted == 2);
    CHECK(wubu_gv_is_admitted(&gv, SYS_READ) == 1);
    CHECK(wubu_gv_is_admitted(&gv, SYS_WRITE) == 0);

    /* install seccomp + ready sentry */
    CHECK(wubu_gv_install_seccomp(&gv) == 0);
    CHECK(gv.seccomp_on == 1);
    CHECK(wubu_gv_install_seccomp(&gv) == 0);  /* idempotent */
    CHECK(wubu_gv_ready_sentry(&gv) == 0);
    CHECK(gv.sentry_ready == 1);
    /* sentry needs seccomp */
    wubu_gvisor_t g2; wubu_gv_init(&g2);
    CHECK(wubu_gv_ready_sentry(&g2) == -1);

    /* service: read and getpid admit; write/open/close deny */
    int act;
    CHECK(wubu_gv_service(&gv, SYS_READ, &act) == 0);
    CHECK(act == WUBU_GV_ACTION_ADMIT);
    CHECK(wubu_gv_service(&gv, SYS_GETPID, &act) == 0);
    CHECK(act == WUBU_GV_ACTION_ADMIT);
    CHECK(wubu_gv_service(&gv, SYS_WRITE, &act) == 0);
    CHECK(act == WUBU_GV_ACTION_DENY);
    CHECK(wubu_gv_service(&gv, SYS_OPEN, &act) == 0);
    CHECK(act == WUBU_GV_ACTION_DENY);
    CHECK(wubu_gv_service(&gv, SYS_CLOSE, &act) == 0);
    CHECK(act == WUBU_GV_ACTION_DENY);
    /* deny counts: write/open/close = 3 denied ; read/getpid admit ; 5 trapped */
    CHECK(gv.n_denied == 3);
    CHECK(wubu_gv_n_trapped(&gv) == 5);   /* read+getpid+write+open+close serviced */

    /* run a stream: read,read,getpid,write(stop) -> 3 admits then DENY aborts */
    int stream1[] = { SYS_READ, SYS_READ, SYS_GETPID, SYS_WRITE };
    int got = wubu_gv_run(&gv, stream1, 4);
    CHECK(got == -1);   /* denied syscall aborts the app */

    /* all-admit stream runs to completion */
    int stream2[] = { SYS_READ, SYS_GETPID, SYS_READ };
    CHECK(wubu_gv_run(&gv, stream2, 3) == 3);

    /* seccomp not installed -> run fails */
    wubu_gvisor_t g3; wubu_gv_init(&g3);
    int s3[] = { SYS_READ };
    CHECK(wubu_gv_run(&g3, s3, 1) == -1);

    /* bad args */
    CHECK(wubu_gv_service(NULL, SYS_READ, &act) == -1);
    CHECK(wubu_gv_admit(NULL, SYS_READ) == -1);
    CHECK(wubu_gv_run(&gv, NULL, 1) == -1);
    CHECK(wubu_gv_is_admitted(&gv, SYS_EPOLL_WAIT) == 0);   /* unregistered */

    if (fails == 0) printf("ALL GVISOR TESTS PASSED\n");
    else printf("%d GVISOR TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
