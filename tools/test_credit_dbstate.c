/* test_credit_dbstate.c -- the credit-assignment SFT mask + the DB-state
 * reward verifier (the Orchard partial-credit + tau-bench stateful-eval
 * doctrine). */
#include <stdio.h>
#include "wubu_credit_sft.h"
#include "wubu_dbstate.h"

int main(void)
{
    int ok = 1;

    /* credit mask: [1 1 0 1] -- leading successes then a failure tail */
    int succ[] = {1, 1, 0, 1};
    int mask[4];
    int c = wubu_credit_mask(succ, 4, mask);
    if (c != 2 || mask[0] != 1 || mask[1] != 1 || mask[2] != 0 || mask[3] != 0) {
        printf("  credit [1 1 0 1] -> %d %d %d %d %d FAIL\n",
               c, mask[0], mask[1], mask[2], mask[3]);
        ok = 0;
    }
    /* all-success: everything credits */
    int succ2[] = {1, 1, 1};
    int mask2[3];
    c = wubu_credit_mask(succ2, 3, mask2);
    if (c != 3 || mask2[2] != 1) { printf("  credit all-1 FAIL\n"); ok = 0; }
    /* first-failure: nothing credits */
    int succ3[] = {0, 1, 1};
    int mask3[3];
    c = wubu_credit_mask(succ3, 3, mask3);
    if (c != 0 || mask3[0] != 0 || mask3[1] != 0 || mask3[2] != 0) {
        printf("  credit first-fail FAIL\n"); ok = 0;
    }
    float f = wubu_credit_frac(mask, 4);
    if (f < 0.49f || f > 0.51f) { printf("  credit frac %.2f FAIL\n", f); ok = 0; }

    /* DB-state: the flight goal {price < 500, carrier == UA} */
    wubu_db_goal_t goals[] = {
        { "price", "<", "500" },
        { "carrier", "==", "UA" },
    };
    wubu_db_slot_t state[] = { { "price", "312" }, { "carrier", "UA" } };
    int r = wubu_db_verify(goals, 2, state, 2);
    if (r != 1) { printf("  db verify met-state -> %d FAIL\n", r); ok = 0; }
    float rw = wubu_db_reward(goals, 2, state, 2);
    if (rw < 0.99f) { printf("  db reward %.2f FAIL\n", rw); ok = 0; }

    /* price violates -> 0 */
    wubu_db_slot_t state2[] = { { "price", "700" }, { "carrier", "UA" } };
    r = wubu_db_verify(goals, 2, state2, 2);
    if (r != 0) { printf("  db verify violated -> %d FAIL\n", r); ok = 0; }
    rw = wubu_db_reward(goals, 2, state2, 2);
    if (rw < 0.49f || rw > 0.51f) { printf("  db partial reward %.2f FAIL\n", rw); ok = 0; }

    /* missing slot -> -1 */
    wubu_db_slot_t state3[] = { { "price", "312" } };
    r = wubu_db_verify(goals, 2, state3, 1);
    if (r != -1) { printf("  db verify missing -> %d FAIL\n", r); ok = 0; }

    /* the string-equality op on a non-numeric goal */
    wubu_db_goal_t g2[] = { { "carrier", "==", "WN" } };
    wubu_db_slot_t s2[] = { { "carrier", "WN" } };
    if (wubu_db_verify(g2, 1, s2, 1) != 1) { printf("  db string-eq FAIL\n"); ok = 0; }

    printf("  credit-assignment + DB-state  %s\n", ok ? "PASS" : "FAIL");
    printf("%s\n", ok ? "ALL CREDIT/DBSTATE TESTS PASSED" : "CREDIT/DBSTATE FAILURES");
    return ok ? 0 : 1;
}
