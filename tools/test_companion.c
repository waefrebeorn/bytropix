/* test_companion.c -- UX #246 Bonzi companion AGI-UX persona.
 * Proves: init idle, mood transitions, expression per mood, tick cadence,
 * no self-transition counted, fail-closed on NULL. */
#include "wubu_companion.h"
#include <stdio.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_companion_t c;
    CHECK(wubu_comp_init(&c) == 0, "init ok");
    CHECK(wubu_comp_mood(&c) == WUBU_COMP_IDLE, "starts idle");
    CHECK(wubu_comp_transitions(&c) == 0, "0 transitions");

    /* idle expression */
    CHECK(strcmp(wubu_comp_expression(&c), "Ready when you are.") == 0, "idle expression");

    /* transition to working */
    CHECK(wubu_comp_set(&c, WUBU_COMP_WORKING) == 0, "set working");
    CHECK(wubu_comp_mood(&c) == WUBU_COMP_WORKING, "mood working");
    CHECK(wubu_comp_transitions(&c) == 1, "1 transition");
    CHECK(strcmp(wubu_comp_expression(&c), "On it...") == 0, "working expression");

    /* setting the SAME mood again is not a new transition */
    CHECK(wubu_comp_set(&c, WUBU_COMP_WORKING) == 0, "set working again");
    CHECK(wubu_comp_transitions(&c) == 1, "still 1 transition (no self-transition)");

    /* ticks accumulate in-mood */
    wubu_comp_tick(&c); wubu_comp_tick(&c); wubu_comp_tick(&c);
    CHECK(c.ticks_in_mood == 3, "3 ticks in working");

    /* transition resets tick count */
    CHECK(wubu_comp_set(&c, WUBU_COMP_CELEBRATE) == 0, "set celebrate");
    CHECK(c.ticks_in_mood == 0, "tick reset on transition");
    CHECK(wubu_comp_transitions(&c) == 2, "2 transitions");
    CHECK(strcmp(wubu_comp_expression(&c), "Goal complete! Nice.") == 0, "celebrate expression");

    /* concern + alert expressions */
    wubu_comp_set(&c, WUBU_COMP_CONCERN);
    CHECK(strcmp(wubu_comp_expression(&c), "Hit a snag -- investigating.") == 0, "concern expression");
    wubu_comp_set(&c, WUBU_COMP_ALERT);
    CHECK(strcmp(wubu_comp_expression(&c), "Heads up -- something needs you.") == 0, "alert expression");

    /* fail-closed */
    CHECK(wubu_comp_init(NULL) == -1, "NULL init -> -1");
    CHECK(wubu_comp_set(&c, 99) == -1, "bad mood -> -1");
    CHECK(wubu_comp_set(NULL, WUBU_COMP_IDLE) == -1, "NULL set -> -1");
    CHECK(wubu_comp_tick(NULL) == -1, "NULL tick -> -1");
    CHECK(wubu_comp_mood(NULL) == WUBU_COMP_IDLE, "NULL mood -> idle");
    CHECK(wubu_comp_transitions(NULL) == 0, "NULL transitions -> 0");

    printf("\n=== test_companion: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
