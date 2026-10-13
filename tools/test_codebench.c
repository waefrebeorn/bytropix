/* test_codebench.c -- PATHS #327 verifiable code generation benchmark.
 * Proves: exec opcodes, task passes only if ALL cases match, pass rate,
 * partial-fail task counts as fail, fail-closed on NULL. */
#include "wubu_codebench.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_codebench_t b;
    CHECK(wubu_cb_init(&b) == 0, "init ok");

    /* exec opcodes */
    CHECK(wubu_cb_exec(0, 5) == 5, "op0: identity");
    CHECK(wubu_cb_exec(1, 5) == 6, "op1: +1");
    CHECK(wubu_cb_exec(2, 5) == 10, "op2: *2");
    CHECK(wubu_cb_exec(3, 4) == 16, "op3: square");
    CHECK(wubu_cb_exec(4, 5) == -5, "op4: negate");

    /* task 1: "increment" (op1), all cases correct */
    wubu_cb_case_t c1[3] = {{1,2},{0,1},{-3,-2}};
    CHECK(wubu_cb_add_task(&b, "increment", 1, c1, 3) == 0, "add increment task");

    /* task 2: "double" (op2), all correct */
    wubu_cb_case_t c2[2] = {{5,10},{0,0}};
    CHECK(wubu_cb_add_task(&b, "double", 2, c2, 2) == 0, "add double task");

    /* task 3: WRONG opcode for its spec (square opcode but expected double) --
     * one case fails -> whole task fails */
    wubu_cb_case_t c3[2] = {{3,6},{4,8}};   /* expects *2 but opcode is square(3) */
    CHECK(wubu_cb_add_task(&b, "buggy", 3, c3, 2) == 0, "add buggy task");

    /* run */
    int cases;
    int passed = wubu_cb_run(&b, &cases);
    CHECK(passed == 2, "2 tasks pass (increment, double); buggy fails");
    CHECK(cases == 7, "7 total cases checked (3+2+2)");
    CHECK(b.last_passed == 2, "last_passed 2");
    /* pass rate = 2/3 */
    double rate = wubu_cb_pass_rate(&b);
    CHECK(rate > 0.66 && rate < 0.67, "pass rate = 2/3");

    /* a single failing case fails the whole task (verifiable: ALL must pass) */
    wubu_codebench_t b2;
    wubu_cb_init(&b2);
    wubu_cb_case_t c4[3] = {{1,2},{2,3},{5,999}};  /* last case wrong */
    wubu_cb_add_task(&b2, "mostly_right", 1, c4, 3);
    int cs;
    CHECK(wubu_cb_run(&b2, &cs) == 0, "task with 1 wrong case fails entirely");

    /* all-pass suite -> rate 1.0 */
    wubu_codebench_t b3;
    wubu_cb_init(&b3);
    wubu_cb_add_task(&b3, "t1", 2, c2, 2);
    wubu_cb_run(&b3, &cs);
    CHECK(wubu_cb_pass_rate(&b3) == 1.0, "all-pass suite -> rate 1.0");

    /* fail-closed */
    CHECK(wubu_cb_init(NULL) == -1, "NULL init -> -1");
    CHECK(wubu_cb_add_task(&b, "x", 0, NULL, 1) == -1, "NULL cases -> -1");
    CHECK(wubu_cb_add_task(&b, "x", 0, c1, 0) == -1, "zero cases -> -1");
    CHECK(wubu_cb_run(NULL, &cs) == -1, "NULL run -> -1");
    CHECK(wubu_cb_pass_rate(NULL) == 0.0, "NULL rate -> 0");

    printf("\n=== test_codebench: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
