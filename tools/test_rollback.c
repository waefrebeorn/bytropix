/* test_rollback.c — AGENT #312: 5+1 rollback on divergence.
 *
 * Measured assertions on the rollback state machine:
 *   1. before any checkpoint, divergence streak never triggers restore (safety).
 *   2. K-1 divergences -> CONVERGE, streak == K-1.
 *   3. K-th divergence WITH a checkpoint -> RESTORE fires, streak resets to 0,
 *      restore count increments.
 *   4. a converged step after arming clears the streak.
 *   5. after a restore, the next K divergences fire again (repeatable cadence).
 *   6. checkpoint with id==0 rejects (no empty restore).
 *   7. streak never exceeds K (restores at exactly K).
 */
#include "wubu_rollback.h"
#include <stdio.h>

static int g_run = 0, g_pass = 0;
#define T(cond, msg) do { \
    g_run++; \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while (0)

int main(void) {
    printf("=== test_rollback (5+1 rollback on divergence, #312) ===\n");

    /* 1: no checkpoint -> divergence never restores. */
    wubu_rollback_t rb;
    wubu_rollback_init(&rb);
    int restored_nockpt = 0;
    for (int i = 0; i < 20; i++)
        if (wubu_rollback_step(&rb, 0) == WUBU_ROLLBACK_RESTORE) restored_nockpt = 1;
    T(restored_nockpt == 0, "no restore before any checkpoint");
    T(wubu_rollback_streak(&rb) == 0, "streak unchanged when never armed");

    /* 2 + 3: arm, diverge K-1 times -> CONVERGE; K-th -> RESTORE, streak reset. */
    wubu_rollback_checkpoint(&rb, 1, (void *)0x42);
    for (int i = 0; i < WUBU_ROLLBACK_K - 1; i++)
        T(wubu_rollback_step(&rb, 0) == WUBU_RB_ACTION_CONVERGE, "K-1 divergences: CONVERGE");
    T(wubu_rollback_streak(&rb) == WUBU_ROLLBACK_K - 1, "streak == K-1 before trigger");
    T(wubu_rollback_step(&rb, 0) == WUBU_ROLLBACK_RESTORE, "K-th divergence: RESTORE fired");
    T(wubu_rollback_streak(&rb) == 0, "streak reset after restore");
    T(wubu_rollback_fires(&rb) == 1, "restore count == 1");

    /* 4: a converged step clears the streak. */
    for (int i = 0; i < 3; i++) wubu_rollback_step(&rb, 0);  /* build streak to 3 */
    T(wubu_rollback_streak(&rb) == 3, "streak 3 before converge");
    wubu_rollback_step(&rb, 1);
    T(wubu_rollback_streak(&rb) == 0, "converge clears streak");

    /* 5: repeatable cadence — next K divergences fire again. */
    for (int i = 0; i < WUBU_ROLLBACK_K; i++) wubu_rollback_step(&rb, 0);
    T(wubu_rollback_fires(&rb) == 2, "second restore fired at K");
    T(wubu_rollback_streak(&rb) == 0, "streak reset again");

    /* 6: id==0 checkpoint rejected. */
    T(wubu_rollback_checkpoint(&rb, 0, NULL) == -1, "id==0 checkpoint rejected");

    /* 7: streak never exceeds K across many steps. */
    int over = 0;
    for (int i = 0; i < 100; i++) {
        wubu_rollback_step(&rb, 0);
        if (wubu_rollback_streak(&rb) > WUBU_ROLLBACK_K) over = 1;
    }
    T(over == 0, "streak never exceeds K");

    printf("=== Results: %d/%d passed ===\n", g_pass, g_run);
    return g_pass == g_run ? 0 : 1;
}
