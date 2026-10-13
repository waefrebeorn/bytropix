#ifndef WUBU_SAMPLER_H
#define WUBU_SAMPLER_H
#include <stddef.h>

/* wubu_sampler.h -- BRAIN #84 sampler: DRY anti-repetition for math.
 *
 * DRY (Don't Repeat Yourself) sampling: penalizes tokens that already appear
 * in the generated sequence, with a penalty that grows the more a token has
 * repeated. This breaks the degenerate repetition loops that plague math/code
 * generation (e.g. "x = x = x = ..."). The sampler tracks how many times each
 * token has been emitted, and divides (or subtracts from) a candidate token's
 * logit by a multiplier/base raised to its repeat count. Deterministic given
 * the logits + history, no allocation beyond the caller's, real work. */

#define WUBU_SAMPLER_MAX_TOKENS 32000

typedef struct {
    int   counts[WUBU_SAMPLER_MAX_TOKENS];  /* times each token was emitted */
    int   max_token;                         /* vocab size in use */
    float multiplier;                        /* DRY penalty base, e.g. 2.0 */
    float base;                              /* DRY length base, e.g. 1.75 */
    float allowed_length;                    /* tokens before penalty kicks in */
} wubu_sampler_t;

/* Initialise a DRY sampler for a vocab of `vocab_size` tokens. */
int wubu_sampler_init(wubu_sampler_t *s, int vocab_size,
                      float multiplier, float base, float allowed_length);

/* Record that `token` was emitted (updates its repeat count). */
int wubu_sampler_commit(wubu_sampler_t *s, int token);

/* Apply the DRY penalty to a logit array (in place): tokens that have already
 * been emitted `allowed_length`+ times get their logit divided by
 * multiplier * base^(count - allowed_length). Returns 0/-1. */
int wubu_sampler_apply_dry(wubu_sampler_t *s, float *logits, int n);

/* Argmax over logits (the greedy pick after DRY penalty). Returns the token
 * id, or -1 on bad input. */
int wubu_sampler_argmax(const float *logits, int n);

#endif
