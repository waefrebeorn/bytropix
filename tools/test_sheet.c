/* test_sheet.c -- Win98/XP Spreadsheet tests. */
#include "wubu_sheet.h"
#include <stdio.h>
#include <math.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)
#define NEAR(a,b) (fabs((a)-(b)) < 1e-6)

int main(void)
{
    wubu_sheet_t sh;
    double v;

    CHECK(wubu_sh_init(&sh, 10, 5) == 0);   /* 10 rows, 5 cols (A-E) */
    CHECK(wubu_sh_init(&sh, 0, 5) == -1);
    CHECK(wubu_sh_init(&sh, 10, 0) == -1);
    CHECK(wubu_sh_init(&sh, 999, 5) == -1);
    CHECK(wubu_sh_init(&sh, 10, 999) == -1);
    CHECK(wubu_sh_init(NULL, 10, 5) == -1);

    /* A1 ref parsing */
    CHECK(wubu_sh_ref(&sh, "A1") == 0);
    CHECK(wubu_sh_ref(&sh, "B1") == 1);
    CHECK(wubu_sh_ref(&sh, "E1") == 4);
    CHECK(wubu_sh_ref(&sh, "A2") == 5);
    CHECK(wubu_sh_ref(&sh, "a1") == 0);     /* case-insensitive col */
    CHECK(wubu_sh_ref(&sh, "F1") == -1);    /* out of cols */
    CHECK(wubu_sh_ref(&sh, "A11") == -1);   /* out of rows */
    CHECK(wubu_sh_ref(&sh, "Z1") == -1);    /* beyond E */
    CHECK(wubu_sh_ref(&sh, "1") == -1);     /* no col */
    CHECK(wubu_sh_ref(&sh, "A") == -1);     /* no row */
    CHECK(wubu_sh_ref(&sh, "") == -1);
    CHECK(wubu_sh_ref(&sh, NULL) == -1);

    /* literals */
    CHECK(wubu_sh_set(&sh, "A1", 10) == 0);
    CHECK(wubu_sh_set(&sh, "B1", 20) == 0);
    CHECK(wubu_sh_get(&sh, "A1", &v) == 0 && NEAR(v, 10));
    CHECK(wubu_sh_get(&sh, "B1", &v) == 0 && NEAR(v, 20));
    CHECK(wubu_sh_set(&sh, "Z1", 1) == -1);   /* bad ref */

    /* formulas: ref op ref */
    CHECK(wubu_sh_set_formula(&sh, "C1", "=A1+B1") == 0);
    CHECK(wubu_sh_get(&sh, "C1", &v) == 0 && NEAR(v, 30));
    CHECK(wubu_sh_set_formula(&sh, "D1", "=A1*B1") == 0);
    CHECK(wubu_sh_get(&sh, "D1", &v) == 0 && NEAR(v, 200));
    CHECK(wubu_sh_set_formula(&sh, "E1", "=B1-A1") == 0);
    CHECK(wubu_sh_get(&sh, "E1", &v) == 0 && NEAR(v, 10));
    CHECK(wubu_sh_set_formula(&sh, "C2", "=B1/A1") == 0);
    CHECK(wubu_sh_get(&sh, "C2", &v) == 0 && NEAR(v, 2));

    /* formula: ref op number, bare ref, bare number */
    CHECK(wubu_sh_set_formula(&sh, "D2", "=A1*2") == 0);
    CHECK(wubu_sh_get(&sh, "D2", &v) == 0 && NEAR(v, 20));
    CHECK(wubu_sh_set_formula(&sh, "E2", "=A1") == 0);
    CHECK(wubu_sh_get(&sh, "E2", &v) == 0 && NEAR(v, 10));
    CHECK(wubu_sh_set_formula(&sh, "A3", "=42") == 0);
    CHECK(wubu_sh_get(&sh, "A3", &v) == 0 && NEAR(v, 42));

    /* chained formula (dependency order) */
    CHECK(wubu_sh_set_formula(&sh, "B3", "=C1+A1") == 0);   /* 30 + 10 = 40 */
    CHECK(wubu_sh_get(&sh, "B3", &v) == 0 && NEAR(v, 40));

    /* division by zero -> error sentinel (err, not circular) */
    CHECK(wubu_sh_set(&sh, "A4", 0) == 0);
    CHECK(wubu_sh_set_formula(&sh, "B4", "=A1/A4") == 0);
    CHECK(wubu_sh_get(&sh, "B4", &v) == 0 && v == WUBU_SH_ERR);
    CHECK(wubu_sh_is_circular(&sh, "B4") == 0);   /* div/0 is an error, not a cycle */

    /* circular reference: A5 = B5, B5 = A5 */
    CHECK(wubu_sh_set_formula(&sh, "A5", "=B5") == 0);
    CHECK(wubu_sh_set_formula(&sh, "B5", "=A5") == 0);
    CHECK(wubu_sh_get(&sh, "A5", &v) == 0 && v == WUBU_SH_ERR);
    CHECK(wubu_sh_is_circular(&sh, "A5") == 1);

    /* self-reference: C5 = C5 + 1 */
    CHECK(wubu_sh_set_formula(&sh, "C5", "=C5+1") == 0);   /* set ok, cycle flagged */
    CHECK(wubu_sh_get(&sh, "C5", &v) == 0 && v == WUBU_SH_ERR);
    CHECK(wubu_sh_is_circular(&sh, "C5") == 1);

    /* bad formulas */
    CHECK(wubu_sh_set_formula(&sh, "A1", "A1+1") == -1);   /* no '=' */
    CHECK(wubu_sh_set_formula(&sh, "A1", NULL) == -1);
    CHECK(wubu_sh_set_formula(&sh, "Z1", "=1") == -1);      /* bad ref */
    CHECK(wubu_sh_set_formula(&sh, "A1", "=A1+") == -1);    /* empty rhs */
    CHECK(wubu_sh_set_formula(&sh, "A1", "=+") == -1);      /* empty lhs */
    CHECK(wubu_sh_set_formula(&sh, "A1", "=ZZ9+1") == -1);  /* bad operand ref */

    /* get bad ref */
    CHECK(wubu_sh_get(&sh, "Z9", &v) == -1);
    CHECK(wubu_sh_get(&sh, "A1", NULL) == -1);
    CHECK(wubu_sh_is_circular(&sh, "Z9") == -1);

    /* non-circular cell is not circular */
    CHECK(wubu_sh_is_circular(&sh, "C1") == 0);

    /* recalc after literal change propagates to dependents */
    CHECK(wubu_sh_set(&sh, "A1", 100) == 0);   /* C1 = A1+B1 = 120 now */
    CHECK(wubu_sh_get(&sh, "C1", &v) == 0 && NEAR(v, 120));

    if (fails == 0) printf("ALL SHEET TESTS PASSED\n");
    else printf("%d SHEET TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
