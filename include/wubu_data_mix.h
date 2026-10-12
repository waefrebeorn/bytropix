#ifndef WUBU_DATA_MIX_H
#define WUBU_DATA_MIX_H
/* wubu_data_mix.h — data mixture weighting (BRAIN #243).
 *
 * Combines `k` data sources with raw weights into normalized mixture
 * proportions p_i = w_i / sum(w). Lets a recipe specify "70% code, 20% prose,
 * 10% math" as weights and get back the sampling probabilities. The forward
 * (wubu_data_mix) is covered by test_data_mix with measured assertions.
 */
#include <stddef.h>

/* Normalize `k` raw weights into out proportions (sum to 1). Returns 0 ok, or
 * -1 on bad args (k<=0, NULL, negative weight, or all-zero weights). */
int wubu_data_mix(const double *weights, int k, double *out_props);

/* Sample a source index from the mixture using a uniform u in [0,1) (the
 * caller's RNG). Returns the chosen index, or -1 on bad args. */
int wubu_data_mix_sample(const double *props, int k, double u);

#endif
