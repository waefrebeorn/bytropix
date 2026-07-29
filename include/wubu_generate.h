/*
 * wubu_generate.h -- autoregressive generation with optional n-gram
 * speculative decoding (doc 018 / K01).
 *
 * WHY (Kevin-Bacon convergence): decode is memory-bandwidth-bound; each
 * autoregressive step is one giant matmul. Speculative decoding (Leviathan et
 * al. 2023) drafts K candidate tokens cheaply (here: from the prompt's own
 * n-gram repetition -- ZERO external model, honoring "no third-party"), verifies
 * them in ONE target forward, and commits the longest consistent prefix plus a
 * bonus token. The accepted tokens are PROVABLY those the target would have
 * emitted (rejection sampling) -- so output is bit-identical to naive decoding,
 * but with fewer forwards. This is the biggest remaining *compute* lever after
 * the KV/weight bandwidth wins (013/014/015).
 *
 * Greedy mode is exactly equivalent: draft tokens are accepted iff they equal
 * the target's argmax at each position; on first mismatch we commit the
 * target argmax. Sampled mode uses rng rejection (wubu_spec_verify_tree).
 */
#ifndef WUBU_GENERATE_H
#define WUBU_GENERATE_H

#include <stdint.h>
#include <stddef.h>
#include "wubu_model.h"
#include "wubu_spec_decode.h"

typedef struct {
    int max_tokens;          /* total tokens to emit (incl. prompt echoes if any) */
    int spec_k;              /* draft depth (0 = no speculative, plain decode) */
    int ngram_order;         /* n-gram context order for the drafter (e.g. 3) */
    int greedy;              /* 1 = argmax sampling, 0 = temperature/rng sample */
    float temperature;       /* used when greedy=0 */
    unsigned int seed;       /* rng seed for sampling */
} wubu_generate_cfg_t;

/* Generate from a prompt of `n_prompt` token ids. Appends emitted tokens to
 * `out` (caller-allocated, >= max_tokens). Returns number of tokens emitted.
 * If `ngram` is NULL, an internal n-gram drafter over the running sequence is
 * used when spec_k > 0. */
int wubu_generate(wubu_model_t *model, const int *prompt, int n_prompt,
                  const wubu_generate_cfg_t *cfg, int *out);

#endif /* WUBU_GENERATE_H */
