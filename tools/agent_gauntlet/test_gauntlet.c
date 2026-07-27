/* test_gauntlet.c -- regression gate for the agent tool gauntlet.
 *
 * Asserts (on the fixture fallback, no 9GB weights needed):
 *  1. gauntlet_edr_init() brings the EDR engine up,
 *  2. all four model slots load (fixture fallback if Colonels absent),
 *  3. gauntlet_run_all() fans a positive number of EDR agent actions,
 *  4. every (model,task) produced a decode (n_actions > 0),
 *  5. the EDR recent-events snapshot returns the fanned actions.
 * Exits 0 on all-pass; non-zero on any assertion failure.
 */
#include "agent_gauntlet.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); return 1; } \
    printf("ok: %s\n", msg); \
} while (0)

int main(void) {
    CHECK(gauntlet_edr_init() == 0, "EDR engine starts");

    int n = gauntlet_load_models();
    CHECK(n == G_N_MODELS, "all four model slots loaded (fixture fallback)");

    GauntletScore scores[G_N_MODELS * G_N_TASKS];
    int total = gauntlet_run_all(scores);
    CHECK(total > 0, "gauntlet fanned EDR agent actions (total>0)");

    int any_actions = 0, any_tool = 0;
    for (int i = 0; i < G_N_MODELS * G_N_TASKS; i++) {
        if (scores[i].n_actions > 0) any_actions++;
        if (scores[i].tool_used) any_tool++;
    }
    CHECK(any_actions == G_N_MODELS * G_N_TASKS,
          "every (model,task) decoded (n_actions>0)");

    EdrEventView ev[64];
    int got = gauntlet_edr_recent(64, ev);
    CHECK(got > 0, "EDR recent-events snapshot returns fanned actions");

    gauntlet_edr_stop();
    printf("\nALL GAUNTLET CHECKS PASSED (models=%d, total_actions=%d, edr_events=%d)\n",
           n, total, got);
    return 0;
}
