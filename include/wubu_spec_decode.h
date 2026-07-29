#ifndef WUBU_SPEC_DECODE_H
#define WUBU_SPEC_DECODE_H

#include <stdint.h>
#include "wubu_ngram.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Tree-draft verification. See wubu_spec_decode.c for semantics.
 * candidates/parent: tree layout (parent-first). parent[i] = parent index or -1.
 * draft_probs[i], target_logits[vocab]: distributions.
 * Returns accepted prefix length; fills accepted[] (caller-allocated, >= n+1). */
int wubu_spec_verify_tree(const int *candidates, const int *parent,
                          const float *draft_probs, const float *target_logits,
                          int n, int vocab, int *accepted, int max_accepted,
                          float rng);

/* MTP bonus-token sampler from residual distribution. Returns token id or -1. */
int wubu_spec_bonus_token(const float *target_logits, const float *draft_probs,
                          int vocab, float rng);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SPEC_DECODE_H */
