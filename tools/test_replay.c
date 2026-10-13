/* test_replay.c -- WORLDSIM #271 determinism replay from seed.
 * Proves: same seed -> same draws, replay reproduces the recorded sequence,
 * verify confirms determinism, different seed -> different sequence,
 * fail-closed on NULL. */
#include "wubu_replay.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    wubu_replay_t r;
    CHECK(wubu_replay_init(&r, 12345) == 0, "init ok (seed 12345)");

    /* generate a sequence of draws */
    uint64_t first[10];
    for (int i = 0; i < 10; i++) first[i] = wubu_replay_draw(&r);
    CHECK(wubu_replay_count(&r) == 10, "10 draws recorded");

    /* re-init with the SAME seed -> same sequence (determinism) */
    wubu_replay_t r2;
    wubu_replay_init(&r2, 12345);
    int same = 1;
    for (int i = 0; i < 10; i++) if (wubu_replay_draw(&r2) != first[i]) same = 0;
    CHECK(same == 1, "same seed -> same 10 draws");

    /* different seed -> different sequence */
    wubu_replay_t r3;
    wubu_replay_init(&r3, 99999);
    int diff = 0;
    for (int i = 0; i < 10; i++) if (wubu_replay_draw(&r3) != first[i]) diff = 1;
    CHECK(diff == 1, "different seed -> different sequence");

    /* replay: begin replay mode and consume the recorded draws */
    CHECK(wubu_replay_begin_replay(&r) == 0, "begin replay ok");
    int replay_match = 1;
    for (int i = 0; i < 10; i++) if (wubu_replay_draw(&r) != first[i]) replay_match = 0;
    CHECK(replay_match == 1, "replay reproduces the recorded sequence exactly");

    /* verify confirms determinism */
    CHECK(wubu_replay_verify(&r) == 1, "verify: deterministic (recorded == fresh LCG run)");

    /* verify requires replay mode */
    wubu_replay_t r4;
    wubu_replay_init(&r4, 1);
    for (int i = 0; i < 5; i++) wubu_replay_draw(&r4);
    CHECK(wubu_replay_verify(&r4) == 0, "verify in generate mode -> 0 (must replay first)");

    /* fail-closed */
    CHECK(wubu_replay_init(NULL, 1) == -1, "NULL init -> -1");
    CHECK(wubu_replay_draw(NULL) == 0, "NULL draw -> 0");
    CHECK(wubu_replay_begin_replay(NULL) == -1, "NULL begin_replay -> -1");
    CHECK(wubu_replay_verify(NULL) == 0, "NULL verify -> 0");
    CHECK(wubu_replay_count(NULL) == 0, "NULL count -> 0");

    printf("\n=== test_replay: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
