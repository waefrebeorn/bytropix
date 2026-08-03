/* wubu_seed.h -- the deterministic environment seed (AC-E13): a
 * splitmix64 PRNG that makes rollouts REPLAYABLE -- the same seed
 * yields the same random sequence and the same shuffle, so a rollout
 * under seed s can be reproduced exactly. */
#ifndef WUBU_SEED_H
#define WUBU_SEED_H

#include <stdint.h>

typedef struct {
    uint64_t state;
} wubu_seed_t;

/* Seed the generator (any 64-bit value; 0 is fine). */
void wubu_seed_init(wubu_seed_t *r, uint64_t seed);

/* The next uniform uint64. */
uint64_t wubu_seed_next(wubu_seed_t *r);

/* A uniform double in [0, 1). */
double wubu_seed_unit(wubu_seed_t *r);

/* Fisher-Yates shuffle in place with the seeded generator. */
void wubu_seed_shuffle(wubu_seed_t *r, int *items, int n);

#endif
