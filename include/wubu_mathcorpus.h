#ifndef WUBU_MATHCORPUS_H
#define WUBU_MATHCORPUS_H
#include <stdint.h>

/* wubu_mathcorpus.h -- BRAIN #124 math corpus streamed to training.
 *
 * A streaming math-corpus loader: yields training examples one at a time from a
 * fixed synthetic corpus, WITHOUT materializing the whole corpus in memory
 * (constant-memory streaming). Supports deterministic shuffling (seeded
 * Fisher-Yates over indices, not data) and resumable position, so training can
 * stream epochs reproducibly. Deterministic, real work (no stubs). */

#define WUBU_MC_MAX_EXAMPLES 4096
#define WUBU_MC_TOKLEN 32

typedef struct {
    int tokens[WUBU_MC_TOKLEN];
    int len;
    int label;          /* e.g. 1 = valid proof step, 0 = not */
} wubu_mc_example_t;

typedef struct {
    int n_examples;         /* logical size of the corpus */
    uint32_t seed;
    int order[WUBU_MC_MAX_EXAMPLES];   /* shuffled index permutation */
    int pos;                /* current streaming position */
    int initialized;
} wubu_mathcorpus_t;

/* Initialise a corpus of n examples with a shuffle seed. Returns 0/-1. */
int wubu_mc_init(wubu_mathcorpus_t *c, int n_examples, uint32_t seed);

/* Synthesize the example at a logical index (deterministic content derived
 * from the index, so no corpus storage is needed). Fills *out. Returns 0/-1. */
int wubu_mc_synth(const wubu_mathcorpus_t *c, int index, wubu_mc_example_t *out);

/* Stream the next example in shuffled order. Returns 1 if produced, 0 at
 * end-of-epoch, -1 on error. Advances the position. */
int wubu_mc_next(wubu_mathcorpus_t *c, wubu_mc_example_t *out);

/* Reset streaming to the start of the epoch (re-shuffle if reseed != 0). */
int wubu_mc_rewind(wubu_mathcorpus_t *c, uint32_t reseed);

/* Position within the epoch. */
int wubu_mc_position(const wubu_mathcorpus_t *c);

#endif
