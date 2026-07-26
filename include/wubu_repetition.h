#ifndef WUBU_REPETITION_H
#define WUBU_REPETITION_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * wubu_repetition.h -- generation-time repetition suppression.
 *
 * Two complementary mechanisms (matching the llama.cpp findings the Colonel
 * flagged for Agents-A1 / Qwen3.6 on RTX 5070 Ti):
 *
 *   1. repeat_penalty  -- penalize tokens that already appear in the
 *      recent window (last `penalty_last_n` tokens). Simple, fast, but
 *      blind to long-range loops (llama.cpp only scans 64 tokens).
 *
 *   2. DRY (Don't Repeat Yourself) -- references the ENTIRE context,
 *      hashes the running suffix against previously-emitted token n-grams,
 *      and damps any token that would extend an already-seen sequence.
 *      This is what kills the 3+ identical-sentence loops.
 *
 * Both operate on the logits IN PLACE before sampling. Opaque state so
 * the caller never touches the rolling buffers.
 */

typedef struct wubu_rep_state wubu_rep_state_t;

// Allocate repetition state.
//   vocab_size:     model vocabulary size (e.g. 248320).
//   penalty_last_n: window for repeat_penalty (<=0 means whole context).
//   dry_ngram_len: max n-gram length DRY matches (e.g. 2).
//   dry_hash_len:   suffix hash window DRY scans (<=0 -> whole context).
wubu_rep_state_t *wubu_rep_create(int vocab_size,
                                      int penalty_last_n,
                                      int dry_ngram_len,
                                      int dry_hash_len);

// Free.
void wubu_rep_free(wubu_rep_state_t *s);

// Configure penalty strengths.
//   repeat_penalty: >1.0 dampens repeats (e.g. 1.05 / 1.1).
//   dry_multiplier: DRY strength (e.g. 0.5 .. 1.2).
//   dry_base:       DRY exponential base (e.g. 1.75).
void wubu_rep_set_params(wubu_rep_state_t *s,
                          float repeat_penalty,
                          float dry_multiplier,
                          float dry_base);

// Roll a freshly-generated token into the context (call AFTER sampling).
void wubu_rep_observe(wubu_rep_state_t *s, int token_id);

// Apply suppression to `logits[vocab_size]` IN PLACE, using the rolling
// context accumulated so far (does NOT consume the current token).
// Returns 0 on success, -1 on misconfiguration.
int wubu_rep_apply(wubu_rep_state_t *s, float *logits);

// Reset context (e.g. for a new sequence / after a rollback).
void wubu_rep_reset(wubu_rep_state_t *s);

#ifdef __cplusplus
}
#endif

#endif // WUBU_REPETITION_H
