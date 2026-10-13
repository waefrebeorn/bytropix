/* test_docverify.c -- STACK #369 docs-vs-code verification.
 * Proves: backed modules counted, drift (documented-but-missing) detected,
 * match rate, consistency flag, fail-closed on NULL. */
#include "wubu_docverify.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_docverify_t v;
    CHECK(wubu_dv_init(&v) == 0, "init ok");

    /* fully backed: docs + code + header */
    wubu_dv_record(&v, "wubu_canary", 1, 1, 1);
    wubu_dv_record(&v, "wubu_islands", 1, 1, 1);
    /* documented but MISSING from code = drift */
    wubu_dv_record(&v, "wubu_ghost", 0, 0, 1);
    /* in code but NOT documented (undocumented, not drift) */
    wubu_dv_record(&v, "wubu_secret", 1, 1, 0);
    /* documented + code but missing header (partial, still drift-free on code) */
    wubu_dv_record(&v, "wubu_nohdr", 1, 0, 1);

    /* backed = docs+code+header all present = canary, islands = 2 */
    CHECK(wubu_dv_backed(&v) == 2, "backed = 2 (canary, islands)");
    /* drift = documented but no code = ghost = 1 */
    CHECK(wubu_dv_drift(&v) == 1, "drift = 1 (ghost)");
    /* documented count = 4 (canary, islands, ghost, nohdr); secret is not */
    /* match rate = backed/documented = 2/4 = 0.5 */
    double rate = wubu_dv_match_rate(&v);
    CHECK(rate > 0.49 && rate < 0.51, "match rate = 0.5 (2 backed / 4 documented)");
    /* not consistent (ghost drift) */
    CHECK(wubu_dv_consistent(&v) == 0, "not consistent (drift present)");

    /* a fully consistent set */
    wubu_docverify_t ok;
    wubu_dv_init(&ok);
    wubu_dv_record(&ok, "a", 1, 1, 1);
    wubu_dv_record(&ok, "b", 1, 1, 1);
    wubu_dv_record(&ok, "c", 1, 1, 0);   /* undocumented code, not drift */
    CHECK(wubu_dv_drift(&ok) == 0, "no drift in consistent set");
    CHECK(wubu_dv_consistent(&ok) == 1, "consistent set -> 1");
    CHECK(wubu_dv_match_rate(&ok) == 1.0, "all documented modules backed -> rate 1.0");

    /* empty set -> vacuously consistent */
    wubu_docverify_t empty;
    wubu_dv_init(&empty);
    CHECK(wubu_dv_drift(&empty) == 0, "empty -> 0 drift");
    CHECK(wubu_dv_match_rate(&empty) == 1.0, "empty -> rate 1.0 (vacuous)");

    /* fail-closed */
    CHECK(wubu_dv_record(&v, "x", 1, 1, 1) == 0, "record ok");
    CHECK(wubu_dv_record(NULL, "x", 1, 1, 1) == -1, "NULL record -> -1");
    CHECK(wubu_dv_drift(NULL) == 0, "NULL drift -> 0");
    CHECK(wubu_dv_backed(NULL) == 0, "NULL backed -> 0");
    CHECK(wubu_dv_match_rate(NULL) == 0.0, "NULL rate -> 0");
    CHECK(wubu_dv_consistent(NULL) == 0, "NULL consistent -> 0 (fail-closed)");

    printf("\n=== test_docverify: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
