/* wubu_replay.c -- WORLDSIM #271 determinism replay from seed.
 * See wubu_replay.h. */
#include "wubu_replay.h"

/* Deterministic LCG (Numerical Recipes constants). */
static uint64_t lcg_next(uint64_t s) {
    return s * 6364136223846793005ULL + 1442695040888963407ULL;
}

int wubu_replay_init(wubu_replay_t *r, uint64_t seed) {
    if (!r) return -1;
    r->seed = seed;
    r->state = seed;
    r->n_draws = 0;
    r->replay_mode = 0;
    r->replay_idx = 0;
    return 0;
}

uint64_t wubu_replay_draw(wubu_replay_t *r) {
    if (!r) return 0;
    r->state = lcg_next(r->state);
    if (r->replay_mode) {
        /* replay: return the recorded draw if in range */
        if (r->replay_idx < r->n_draws) {
            return r->draws[r->replay_idx++];
        }
        return r->state;   /* beyond recording: keep generating deterministically */
    }
    /* generate: record */
    if (r->n_draws < WUBU_REPLAY_MAX) {
        r->draws[r->n_draws++] = r->state;
    }
    return r->state;
}

int wubu_replay_begin_replay(wubu_replay_t *r) {
    if (!r) return -1;
    r->state = r->seed;   /* rewind to seed */
    r->replay_mode = 1;
    r->replay_idx = 0;
    return 0;
}

int wubu_replay_verify(wubu_replay_t *r) {
    if (!r) return 0;
    if (!r->replay_mode) return 0;   /* must be in replay mode */
    /* replay consumed draws; verify a fresh LCG run from seed reproduces them */
    uint64_t s = r->seed;
    for (int i = 0; i < r->n_draws; i++) {
        s = lcg_next(s);
        if (s != r->draws[i]) return 0;   /* mismatch -> not deterministic */
    }
    return 1;
}

int wubu_replay_count(const wubu_replay_t *r) {
    return r ? r->n_draws : 0;
}
