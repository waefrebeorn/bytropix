/* test_user_sim.c -- the tau-bench-style user simulator: the react /
 * verify / utter state machine against the goal constraints. */
#include <stdio.h>
#include <string.h>
#include "wubu_user_sim.h"

int main(void)
{
    /* scenario: the user wants a flight under $500 (terse persona) */
    wubu_us_user_t u = { "terse-flier", "price", "<", 500.0, 0, NULL,
                         "flight under $500" };
    wubu_us_slot_t state[4];
    int ok = 1;

    /* the agent's first action: a flight found at $600 (policy violation) */
    state[0] = (wubu_us_slot_t){ "price", "600" };
    int r = wubu_us_react(&u, state, 1);
    if (r != -1) { printf("  price=600 should push back, got %d FAIL\n", r); ok = 0; }
    char out[128];
    wubu_us_utter(&u, state, 1, out, sizeof out);
    if (strcmp(out, "that's not what I asked for") != 0) {
        printf("  terse pushback utterance '%s' FAIL\n", out); ok = 0;
    }

    /* the agent books a $312 flight: goal met */
    state[0] = (wubu_us_slot_t){ "price", "312" };
    r = wubu_us_react(&u, state, 1);
    if (r != 1) { printf("  price=312 should satisfy, got %d FAIL\n", r); ok = 0; }
    if (!wubu_us_verify(&u, state, 1)) { printf("  verify 312 FAIL\n"); ok = 0; }

    /* the slot missing: keep going */
    r = wubu_us_react(&u, NULL, 0);
    if (r != 0) { printf("  missing state should keep going, got %d FAIL\n", r); ok = 0; }

    /* a verbose persona with a >= goal */
    wubu_us_user_t v = { "verbose-manager", "rating", ">=", 4.0, 1, NULL,
                         "at least a 4-star rating" };
    state[0] = (wubu_us_slot_t){ "rating", "3.5" };
    r = wubu_us_react(&v, state, 1);
    if (r != -1) { printf("  rating 3.5 < 4 should push back, got %d FAIL\n", r); ok = 0; }
    wubu_us_utter(&v, state, 1, out, sizeof out);
    if (strstr(out, "does not meet") == NULL) {
        printf("  verbose pushback '%s' FAIL\n", out); ok = 0;
    }
    state[0] = (wubu_us_slot_t){ "rating", "4.5" };
    if (!wubu_us_verify(&v, state, 1)) { printf("  verify 4.5 FAIL\n"); ok = 0; }

    /* the exact-value goal (non-numeric slot): the carrier must be UA */
    wubu_us_user_t e = { "brand-loyal", "carrier", "==", 0.0, 0, "UA", NULL };
    state[0] = (wubu_us_slot_t){ "carrier", "WN" };
    r = wubu_us_react(&e, state, 1);
    if (r != -1) { printf("  carrier WN should fail exact-UA, got %d FAIL\n", r); ok = 0; }
    state[0] = (wubu_us_slot_t){ "carrier", "UA" };
    r = wubu_us_react(&e, state, 1);
    if (r != 1) { printf("  carrier UA should satisfy exact-UA, got %d FAIL\n", r); ok = 0; }

    printf("  react/verify/utter state machine  %s\n", ok ? "PASS" : "FAIL");
    printf("%s\n", ok ? "ALL USER-SIM TESTS PASSED" : "USER-SIM FAILURES");
    return ok ? 0 : 1;
}
