#ifndef WUBU_TOKEN_DROPOUT_H
#define WUBU_TOKEN_DROPOUT_H
/* wubu_token_dropout.h — token dropout (BRAIN #240).
 *
 * Randomly zeroes each token id with probability `p`, mapping dropped tokens
 * to a sentinel `mask_id` and keeping the rest (inverted dropout scales the
 * kept logits, but for token IDs the standard is: keep with prob 1-p, else
 * mask_id). The forward (wubu_token_dropout) is covered by test_token_dropout
 * with measured assertions on the drop rate and mask sentinel.
 *
 * A caller-supplied xorshift RNG makes the forward deterministic in tests.
 */
#include <stddef.h>
#include <stdint.h>

/* A tiny deterministic xorshift64* RNG state. */
typedef struct { uint64_t s; } wubu_rng_t;

/* Seed the RNG (s==0 is remapped to a nonzero constant). */
void wubu_rng_seed(wubu_rng_t *r, uint64_t s);

/* Next uniform double in [0,1). */
double wubu_rng_next(wubu_rng_t *r);

/* Apply token dropout in place to `n` token ids: with probability `p`, replace
 * id with mask_id; else keep. Returns the number of tokens dropped, or -1 on
 * bad args (n<=0, NULL, p<0, p>1). */
int wubu_token_dropout(int *tokens, int n, double p, int mask_id, wubu_rng_t *rng);

#endif
