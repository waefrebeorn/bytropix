/* gauntlet_run.c -- CLI driver for the bytropix agent tool gauntlet.
 *
 * Loads all four Colonel models (fixture fallback), runs them through the
 * agent-tool battery, fans every agent action into the WuBuOS EDR layer,
 * and prints a per-model leaderboard. This is the OS AGI self-improvement
 * loop's evidence generator.
 */
#include "agent_gauntlet.h"
#include <stdio.h>

static const char *task_name(int t) {
    return t == 0 ? "shell" : t == 1 ? "file" : "code";
}

int main(void) {
    printf("== bytropix agent tool gauntlet (4 models x 3 tools, EDR fan-out) ==\n");
    gauntlet_edr_init();

    int n = gauntlet_load_models();
    printf("loaded %d/%d models\n", n, G_N_MODELS);

    GauntletScore scores[G_N_MODELS * G_N_TASKS];
    int total = gauntlet_run_all(scores);
    printf("total EDR agent actions fanned: %d\n", total);

    /* Leaderboard: correctness score per model = sum of correct across tasks. */
    printf("\n%-18s %8s %8s %8s %10s\n", "MODEL", "shell", "file", "code", "score");
    for (int i = 0; i < G_N_MODELS; i++) {
        int s = 0;
        int cell[3] = {0,0,0};
        for (int t = 0; t < G_N_TASKS; t++) {
            GauntletScore *sc = &scores[i * G_N_TASKS + t];
            cell[t] = sc->correct;
            s += sc->correct;
        }
        printf("%-18s %8d %8d %8d %10d\n",
               g_models[i].loaded ? g_models[i].name : "(unloaded)",
               cell[0], cell[1], cell[2], s);
    }

    /* Replay snapshot from EDR (what the OS self-improvement loop consumes). */
    EdrEventView ev[16];
    int got = gauntlet_edr_recent(16, ev);
    printf("\nEDR recent events (audit tail): %d\n", got);
    for (int i = 0; i < got; i++)
        printf("  [%d] type=%u detail='%s'\n", i, ev[i].type, ev[i].detail);

    gauntlet_edr_stop();
    printf("\nDONE.\n");
    return 0;
}
