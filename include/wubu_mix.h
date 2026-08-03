/* wubu_mix.h -- the weighted multi-stream corpus mixer (the corpus-mix
 * recipe as code): the training corpus is the weighted blend of the
 * staged streams -- the pretraining stream (FineMath/OpenMath/cosmopedia)
 * with the per-stream weights from the 045 recipe. The sampler is a
 * deterministic weighted round-robin: each chunk picks the stream i with
 * probability proportional to its weight. */
#ifndef WUBU_MIX_H
#define WUBU_MIX_H

#include <stdint.h>

/* Build the mixed stream into out (at most out_cap tokens).
 * paths [n]: the .tok file paths; weights [n]: the sampling weights.
 * chunk: the round-robin chunk size (e.g. 65536).
 * Returns the number of tokens written, or -1 on an error. */
long wubu_mix_build(const char **paths, const float *weights, int n,
                    uint16_t *out, long out_cap, int chunk);

#endif
