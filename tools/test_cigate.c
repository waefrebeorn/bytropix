/* test_cigate.c -- STACK #368 tests: three-repo integration CI gate.
 *
 * Exercises the gate against REAL repo commands: the wubunos compiler gate
 * (gauntlet_runner on a passing and a failing program), plus green/red rollup.
 * The full three-repo command set is provided via wubu_ci_run_leg; the test
 * uses bounded single-case gates so it stays fast. */

#include "wubu_cigate.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_ci_report_t r;
    wubu_ci_leg_t leg;
    char summary[128];

    /* --- a trivially passing command --- */
    CHECK(wubu_ci_init(&r) == 0);
    CHECK(wubu_ci_run_leg(&leg, "trivial", "true") == 0);
    CHECK(leg.ran == 1);
    CHECK(leg.exit_code == 0);
    CHECK(leg.passed == 1);

    /* --- a trivially failing command --- */
    CHECK(wubu_ci_run_leg(&leg, "failer", "false") == 0);
    CHECK(leg.ran == 1);
    CHECK(leg.exit_code != 0);
    CHECK(leg.passed == 0);

    /* --- output capture (tail) --- */
    CHECK(wubu_ci_run_leg(&leg, "echo", "printf 'HELLO_MARKER_XYZ\\n'") == 0);
    CHECK(strstr(leg.output, "HELLO_MARKER_XYZ") != NULL);

    /* --- large output truncation keeps the tail, not the head --- */
    CHECK(wubu_ci_run_leg(&leg, "big", "for i in $(seq 1 5000); do echo line_$i; done") == 0);
    CHECK(strlen(leg.output) < WUBU_CI_OUT_MAX);
    CHECK(strstr(leg.output, "line_5000") != NULL);   /* tail preserved */

    /* --- nonexistent command -> child exits 127, leg fails --- */
    CHECK(wubu_ci_run_leg(&leg, "nocmd", "/no/such/binary_xyz") == 0);
    CHECK(leg.passed == 0);

    /* --- real compiler gate (wubunos): a passing program --- */
    {
        const char *cmd =
            "cd /home/wubu/wubunos && ./gauntlet_runner --dump-asm x86-64 "
            "'int main(void){ return 0; }' 2>&1 | grep -q 'result=0'";
        CHECK(wubu_ci_run_leg(&leg, "compiler", cmd) == 0);
        CHECK(leg.ran == 1);
        /* the gate passes iff gauntlet_runner reports result=0 */
        CHECK(leg.passed == 1);
    }

    /* --- real compiler gate (wubunos): a failing program --- */
    {
        const char *cmd =
            "cd /home/wubu/wubunos && ./gauntlet_runner --dump-asm x86-64 "
            "'int main(void){ return 12345; }' 2>&1 | grep -q 'result=0'";
        CHECK(wubu_ci_run_leg(&leg, "compiler", cmd) == 0);
        CHECK(leg.passed == 0);   /* result=12345, not 0 -> grep fails */
    }

    /* --- real brain gate (wubuwizard): a passing test binary --- */
    {
        const char *cmd =
            "cd /home/wubu/wubuwizard && make -s test_nvmeqd >/dev/null 2>&1; "
            "./test_nvmeqd 2>&1 | grep -q 'ALL NVMeQD TESTS PASSED'";
        CHECK(wubu_ci_run_leg(&leg, "brain", cmd) == 0);
        CHECK(leg.passed == 1);
    }

    /* --- rollup: all green --- */
    {
        CHECK(wubu_ci_init(&r) == 0);
        wubu_ci_run_leg(&r.legs[0], "compiler", "true");
        r.legs[0].passed = 1;
        wubu_ci_run_leg(&r.legs[1], "brain", "true");
        r.legs[1].passed = 1;
        wubu_ci_run_leg(&r.legs[2], "kernel", "true");
        r.legs[2].passed = 1;
        r.n_legs = 3;
        wubu_ci_finalize(&r);
        CHECK(r.n_passed == 3);
        CHECK(r.n_failed == 0);
        CHECK(r.verdict == WUBU_CI_GREEN);
        wubu_ci_summary(&r, summary, sizeof summary);
        CHECK(strstr(summary, "GREEN") != NULL);
        CHECK(strstr(summary, "3/3") != NULL);
    }

    /* --- rollup: one red --- */
    {
        CHECK(wubu_ci_init(&r) == 0);
        wubu_ci_run_leg(&r.legs[0], "compiler", "true"); r.legs[0].passed = 1;
        wubu_ci_run_leg(&r.legs[1], "brain", "false");  r.legs[1].passed = 0;
        wubu_ci_run_leg(&r.legs[2], "kernel", "true");  r.legs[2].passed = 1;
        r.n_legs = 3;
        wubu_ci_finalize(&r);
        CHECK(r.n_passed == 2);
        CHECK(r.n_failed == 1);
        CHECK(r.verdict == WUBU_CI_RED);
        wubu_ci_summary(&r, summary, sizeof summary);
        CHECK(strstr(summary, "RED") != NULL);
        CHECK(strstr(summary, "2/3") != NULL);
    }

    /* --- empty report is RED (no legs) --- */
    {
        CHECK(wubu_ci_init(&r) == 0);
        r.n_legs = 0;
        wubu_ci_finalize(&r);
        CHECK(r.verdict == WUBU_CI_RED);
    }

    if (fails == 0) printf("ALL CIGATE TESTS PASSED\n");
    else printf("%d CIGATE TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
