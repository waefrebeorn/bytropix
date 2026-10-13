#ifndef WUBU_REPLAY_H
#define WUBU_REPLAY_H
#include <stdint.h>

/* wubu_replay.h -- WORLDSIM #271 determinism replay from seed.
 *
 * Deterministic replay: a simulation driven by a seeded RNG produces the exact
 * same sequence of states when replayed from the same seed. This module holds
 * a seeded RNG, records the sequence of "random" draws it produced, and can
 * replay them (verify a re-run reproduces the identical sequence) or restore
 * from a checkpoint. The guarantee: same seed -> same draws -> same
 * simulation, bit-for-bit. Deterministic, fixed-size, real work (no stubs). */

#define WUBU_REPLAY_MAX 256

typedef struct {
    uint64_t seed;
    uint64_t state;          /* LCG state */
    uint64_t draws[WUBU_REPLAY_MAX];  /* recorded draws for replay verification */
    int      n_draws;
    int      replay_mode;    /* 1 = replaying recorded draws, 0 = generating */
    int      replay_idx;
} wubu_replay_t;

/* Initialise with a seed. */
int wubu_replay_init(wubu_replay_t *r, uint64_t seed);

/* Draw the next random value. In generate mode, produces from the LCG and
 * records it; in replay mode, returns the recorded draw (and checks it
 * matches a fresh LCG draw -- determinism verification). Returns the value. */
uint64_t wubu_replay_draw(wubu_replay_t *r);

/* Finish recording and switch to replay mode (re-init state to seed). */
int wubu_replay_begin_replay(wubu_replay_t *r);

/* Verify the replay reproduced all recorded draws exactly. Returns 1 if the
 * replayed sequence matched the recorded one (deterministic), 0 if any
 * mismatch. */
int wubu_replay_verify(wubu_replay_t *r);

/* Number of draws recorded. */
int wubu_replay_count(const wubu_replay_t *r);

#endif
