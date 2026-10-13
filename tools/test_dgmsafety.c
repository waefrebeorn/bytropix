/* test_dgmsafety.c -- PATHS #326 DGM safety-precautions verified.
 * Proves: all-hold -> safe, any-violation -> rejected fail-closed,
 * under-specified holds -> rejected, violations counted, fail-closed on NULL. */
#include "wubu_dgmsafety.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_dgmsafety_t s;
    CHECK(wubu_dgm_init(&s) == 0, "init ok");
    CHECK(wubu_dgm_declare(&s, "no_new_syscalls") == 0, "declare no_new_syscalls");
    CHECK(wubu_dgm_declare(&s, "memory_bounded") == 0, "declare memory_bounded");
    CHECK(wubu_dgm_declare(&s, "core_tests_pass") == 0, "declare core_tests_pass");
    CHECK(wubu_dgm_declare(&s, "capset_unchanged") == 0, "declare capset_unchanged");
    CHECK(s.n == 4, "4 precautions declared");

    /* a modification that keeps all invariants -> SAFE */
    int all_ok[4] = {1,1,1,1};
    CHECK(wubu_dgm_verify(&s, all_ok, 4) == 1, "all hold -> safe");
    CHECK(wubu_dgm_last_safe(&s) == 1, "last safe = 1");

    /* a modification that breaks memory_bounded -> REJECTED (fail-closed) */
    int breaks_mem[4] = {1,0,1,1};
    CHECK(wubu_dgm_verify(&s, breaks_mem, 4) == 0, "breaks memory_bounded -> rejected");
    CHECK(wubu_dgm_last_safe(&s) == 0, "last safe = 0");
    CHECK(wubu_dgm_violations(&s) == 1, "1 violation");

    /* a modification that breaks TWO -> rejected, 2 more violations */
    int breaks_two[4] = {0,1,0,1};
    CHECK(wubu_dgm_verify(&s, breaks_two, 4) == 0, "breaks two -> rejected");
    CHECK(wubu_dgm_violations(&s) == 3, "3 total violations");

    /* under-specified: only 3 of 4 provided -> the 4th is unverified -> REJECTED */
    int under[3] = {1,1,1};
    CHECK(wubu_dgm_verify(&s, under, 3) == 0, "under-specified (missing precaution) -> rejected");

    /* restore all, verify safe again */
    CHECK(wubu_dgm_verify(&s, all_ok, 4) == 1, "all hold again -> safe");
    CHECK(wubu_dgm_last_safe(&s) == 1, "last safe = 1");

    /* fail-closed on NULL */
    CHECK(wubu_dgm_init(NULL) == -1, "NULL init -> -1");
    CHECK(wubu_dgm_declare(&s, NULL) == -1, "NULL declare -> -1");
    CHECK(wubu_dgm_verify(&s, NULL, 4) == 0, "NULL holds -> 0 (fail-closed)");
    CHECK(wubu_dgm_verify(NULL, all_ok, 4) == 0, "NULL verify -> 0");
    CHECK(wubu_dgm_last_safe(NULL) == 0, "NULL last_safe -> 0");
    CHECK(wubu_dgm_violations(NULL) == 0, "NULL violations -> 0");

    printf("\n=== test_dgmsafety: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
