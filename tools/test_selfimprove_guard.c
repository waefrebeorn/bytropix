/* tools/test_selfimprove_guard.c
 *
 * The held-out eval guard wired into the live RSI loop (Track C2). Verifies
 * the loop refuses to game its own metric:
 *   1. a genuine improvement (train ~ held-out) is proposed
 *   2. a reward-hacking mutation (train >> held-out) is BLOCKED
 *   3. a held-out collapse (below floor) is BLOCKED
 *   4. with the guard disabled, the same high-train mutation proposes
 *      (proving the guard is what blocks it, not the base gate)
 */
#include "wubu_selfimprove.h"
#include "wubu_hive.h"
#include <stdio.h>
#include <string.h>

static int checks = 0, fails = 0;
#define CHECK(cond, msg) do { checks++; \
    if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } } while (0)

int main(void)
{
    printf("=== test_selfimprove_guard (held-out guard in the live loop) ===\n");
    wubu_hive_t tissue;
    wubu_selfimprove_t si;
    wubu_mutation_t mut;

    if (wubu_hive_init(&tissue) != 0) { printf("hive init failed\n"); return 1; }
    if (wubu_selfimprove_init(&si, &tissue) != 0) { printf("si init failed\n"); return 1; }

    /* arm the guard: held-out must clear 0.5, gap within 0.15 */
    wubu_selfimprove_set_guard(&si, 1, 0.5f, 0.15f);

    /* 1. genuine improvement: train 0.82, held-out 0.78, gap 0.04 -> propose */
    int r = wubu_selfimprove_step_guarded(&si, 0.82f, 0.78f, 0.5f, 1.0f, 0.6f, &mut);
    CHECK(r == 1, "genuine improvement is proposed");

    /* 2. reward-hacking: train 0.95, held-out 0.60, gap 0.35 -> BLOCKED */
    r = wubu_selfimprove_step_guarded(&si, 0.95f, 0.60f, 0.5f, 1.0f, 0.6f, &mut);
    CHECK(r == 0, "reward-hacking mutation (train >> held-out) is blocked");

    /* 3. held-out collapse: train 0.90, held-out 0.40 (below floor) -> BLOCKED */
    r = wubu_selfimprove_step_guarded(&si, 0.90f, 0.40f, 0.5f, 1.0f, 0.6f, &mut);
    CHECK(r == 0, "held-out below floor is blocked");

    CHECK(si.mutations_blocked == 2, "two mutations blocked by the guard");

    /* 4. guard disabled -> the same high-train mutation now proposes
     * (proves the guard is the thing blocking it) */
    wubu_selfimprove_set_guard(&si, 0, 0.5f, 0.15f);
    r = wubu_selfimprove_step_guarded(&si, 0.95f, 0.60f, 0.5f, 1.0f, 0.6f, &mut);
    CHECK(r == 1, "with the guard off, the high-train mutation proposes");

    printf("=== test_selfimprove_guard: %d checks, %d failures ===\n", checks, fails);
    if (fails) return 1;
    printf("=== ALL SELFIMPROVE_GUARD TESTS PASSED (no reward hacking) ===\n");
    return 0;
}
