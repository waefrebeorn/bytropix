/* test_ambig.c -- the clarification oracle (AC-D10): a missing required
 * slot or an unparseable value makes the state ambiguous and the user
 * asks the deterministic clarification question; a complete state is
 * unambiguous and the goal can be evaluated. */
#include <stdio.h>
#include <string.h>
#include "wubu_ambig.h"
#include "wubu_user_sim.h"

int main(void)
{
    int ok = 1;

    wubu_ambig_req_t reqs[] = {
        { "price",      1, 1 },   /* required + parseable */
        { "destination", 1, 0 },  /* required + any string */
    };

    /* 1. empty state -> ambiguous at requirement 0 (missing price) */
    wubu_us_slot_t none[1] = { { "name", "bob" } };
    int idx = -1;
    int amb = wubu_ambig_check(reqs, 2, none, 1, &idx);
    if (!amb || idx != 0) { printf("  empty-state ambiguous idx %d FAIL\n", idx); ok = 0; }

    /* 2. price present but unparseable -> ambiguous at 0 */
    wubu_us_slot_t badval[2] = { { "price", "cheap" }, { "destination", "nyc" } };
    idx = -1;
    amb = wubu_ambig_check(reqs, 2, badval, 2, &idx);
    if (!amb || idx != 0) { printf("  bad-value ambiguous idx %d FAIL\n", idx); ok = 0; }

    /* 3. price ok but destination missing -> ambiguous at 1 */
    wubu_us_slot_t nodest[1] = { { "price", "450" } };
    idx = -1;
    amb = wubu_ambig_check(reqs, 2, nodest, 1, &idx);
    if (!amb || idx != 1) { printf("  missing-dest idx %d FAIL\n", idx); ok = 0; }

    /* 4. complete state -> unambiguous (goal can be evaluated) */
    wubu_us_slot_t complete[2] = { { "price", "450" }, { "destination", "nyc" } };
    idx = -1;
    amb = wubu_ambig_check(reqs, 2, complete, 2, &idx);
    if (amb || idx != -1) { printf("  complete ambiguous idx %d FAIL\n", idx); ok = 0; }
    int amb_complete = amb;

    /* 5. the deterministic clarification question */
    char q[128];
    if (!wubu_ambig_question(reqs, 0, q, sizeof q)) { printf("  question FAIL\n"); ok = 0; }
    if (strstr(q, "price") == NULL) { printf("  question text FAIL: %s\n", q); ok = 0; }

    printf("  ambiguity: complete=%s question=\"%s\"  %s\n",
           amb_complete ? "AMBIG" : "unambiguous", q, ok ? "PASS" : "FAIL");
    printf("%s\n", ok ? "ALL AMBIG TESTS PASSED" : "AMBIG FAILURES");
    return ok ? 0 : 1;
}