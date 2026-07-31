/*
 * wubu_spec_decode.h — Speculative decoding framework for wubuwizard
 *
 * Enables lossless token generation acceleration via a lightweight
 * draft model that proposes K tokens ahead, verified in a single
 * parallel forward pass of the target model. Accepts the longest
 * matching prefix via rejection sampling.
 *
 * C11 zero-malloc design. Opaque context. Self-contained.
 */
#ifndef WUBU_SPEC_DECODE_H
#define WUBU_SPEC_DECODE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_spec_decode_ctx wubu_spec_decode_ctx_t;

/* Initialize speculative decoding context.
 * draft_vocab_size:  vocabulary size of the draft model
 * target_vocab_size: vocabulary size of the target model
 * max_draft_len:     maximum tokens the draft proposes (typically 1-8)
 * Returns NULL on OOM. */
wubu_spec_decode_ctx_t *wubu_spec_decode_init(
        int draft_vocab_size, int target_vocab_size, int max_draft_len);

void wubu_spec_decode_free(wubu_spec_decode_ctx_t *ctx);

/* Speculative decode step: draft proposes tokens, target verifies.
 *
 * q_logits:    [target_vocab_size] — target model logits for current position
 * draft_logit_batch: [max_draft_len][draft_vocab_size] — draft model logits
 *                    for each proposed token position
 * accepted:    output — indices of accepted draft tokens (0..n_accepted-1)
 * n_accepted:  output — number of tokens accepted (0..max_draft_len)
 * n_rejected:  output — number of tokens rejected (always 0 or 1 in
 *                    standard SD+rejection sampling)
 * seed:        PRNG seed for rejection sampling (useful for CUDA/OpenMP)
 *
 * Algorithm:
 * 1. Compute acceptance probability: α_i = min(1, P_target(t_i) / P_draft(t_i))
 * 2. For each draft position i, roll RNG against α_i
 * 3. Accept tokens until first rejection; that token is also accepted
 *    if its target prob ≥ draft prob (rejection sampling variant)
 * 4. Return n_accepted accepted token indices
 *
 * After acceptance, the caller advances the target model by n_accepted
 * tokens and calls wubu_spec_decode() again from the new position. */
int wubu_spec_decode(
        wubu_spec_decode_ctx_t *ctx,
        const float *q_logits,
        const float *draft_logit_batch,
        int *accepted,
        int *n_accepted,
        int *n_rejected,
        uint64_t seed);

/* Compute expected throughput gain given acceptance rate.
 * n_draft_tokens: number of tokens the draft proposes per iteration
 * accept_rate: probability that a draft token is accepted (0..1)
 * Returns: expected tokens generated per target forward pass */
float wubu_spec_decode_throughput(int n_draft_tokens, float accept_rate);

/* EAGLE-3 style: use draft model output as additional conditioning
 * for target model's next-token prediction, increasing acceptance rate.
 * draft_states: [max_draft_len][draft_hidden_dim] — hidden states from draft
 * target_cond: [target_hidden_dim] — updated conditioning vector for target */
void wubu_spec_decode_eagle3_conditioning(
        const float *draft_logit_batch,
        const float *draft_states,
        int max_draft_len,
        int draft_vocab_size,
        float *target_cond,
        float temperature);

/* Tree-based speculative verification (Medusa/EAGLE pattern).
 *
 * candidates:  [n_cand] — candidate token IDs in the draft tree
 * parents:     [n_cand] — parent index for each candidate (-1 = root)
 * draft_probs: [n_cand] — draft model's probability for each candidate
 * target_probs:[vocab]  — target model's probability distribution
 * n_cand:      number of candidates in the tree
 * vocab:       vocabulary size
 * accepted:    output — accepted token IDs (up to n_cand)
 * max_acc:     max accepted tokens to write
 * rng_val:     RNG value in [0,1) for rejection sampling
 *
 * Returns: number of accepted tokens (0..n_cand).
 * Algorithm: walk the tree in BFS order, accept if p_target >= p_draft
 *            or RNG < p_target/p_draft. Stop at first rejection. */
int wubu_spec_verify_tree(
        const int *candidates, const int *parents,
        const float *draft_probs, const float *target_probs,
        int n_cand, int vocab,
        int *accepted, int max_acc, float rng_val);

/* Bonus token: sample from residual distribution (target - draft).
 * Used when all draft tokens are accepted to get one extra free token.
 *
 * target_probs: [vocab] — target model's probability distribution
 * draft_probs:  [vocab]  — draft model's probability distribution (same vocab)
 * vocab:        vocabulary size
 * rng_val:      RNG value in [0,1) for sampling
 *
 * Returns: sampled token ID from residual distribution. */
int wubu_spec_bonus_token(
        const float *target_probs, const float *draft_probs,
        int vocab, float rng_val);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_SPEC_DECODE_H */
