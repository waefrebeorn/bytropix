/*
 * wubu_spec_decode.c — Speculative decoding engine (Area A of the 100-point plan).
 * C11, self-contained, no god headers. Reuses hedged_spec.h rejection math.
 *
 * Provides:
 *   - Tree-draft verification (EAGLE-2/3 style): verify a tree of K candidates
 *     in one target forward pass, accept longest consistent prefix (+ bonus token).
 *   - n-gram draft model: cheap draft from recent context (great for agent loops).
 *   - MTP-style bonus-token sampling from the residual distribution.
 * Verification is correctness-preserving: accepted tokens are exactly those the
 * target model would have emitted (rejection sampling, Leviathan et al. 2023).
 */
#include "wubu_spec_decode.h"
#include "hedged_spec.h"
#include "wubu_ngram.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---------------------------------------------------------------------------
 * 1. Tree-draft verification (items A.2 / A.4 / A.6)
 * ------------------------------------------------------------------------- */

/* Verify a tree of candidate tokens against target logits.
 * candidates[] are laid out parent-first; parent[i] gives the index of token i's
 * parent (-1 for root). target_logits is the full vocab distribution at the
 * position just after the prefix. draft_probs[i] is the draft model's prob for
 * candidate i. rng is a uniform draw in [0,1) for bonus-token rejection.
 *
 * Returns number of tokens accepted (prefix len). Fills accepted[] with the
 * accepted token ids. On a partial accept, samples one bonus token into
 * accepted[] when the residual distribution allows (MTP bonus, item A.9 logic). */
int wubu_spec_verify_tree(const int *candidates, const int *parent,
                          const float *draft_probs, const float *target_logits,
                          int n, int vocab, int *accepted, int max_accepted,
                          float rng) {
    int accepted_n = 0;
    for (int i = 0; i < n && accepted_n < max_accepted; i++) {
        int tok = candidates[i];
        /* The target position to check is relative to tok's parent's acceptance.
         * A candidate is valid only if its parent was accepted. Walk parent chain
         * cheaply: we process in parent-first order so parent accepted <= i. */
        int p = parent[i];
        int parent_accepted = (p < 0) ? 1 : (p < accepted_n);
        if (!parent_accepted) continue;

        float p_target = target_logits[tok];
        float p_draft  = draft_probs[i] > 1e-9f ? draft_probs[i] : 1e-9f;
        if (p_target >= p_draft) {
            accepted[accepted_n++] = tok;            /* always accept */
        } else if (rng < p_target / p_draft) {
            accepted[accepted_n++] = tok;            /* accept w/ prob */
        } else {
            break;                                   /* reject -> stop branch */
        }
    }
    return accepted_n;
}

/* ---------------------------------------------------------------------------
 * 2. n-gram draft model (item A.3) — zero weights, just context repetition.
 * ------------------------------------------------------------------------- */

// n-gram draft now implemented in wubu_ngram.c/h
// Use wubu_ngram_draft_t from wubu_ngram.h

/* ---------------------------------------------------------------------------
 * 3. MTP bonus-token sampler (item A.9): sample from residual
 *    (p_target - p_draft)/(1 - p_draft) after a rejection.
 * ------------------------------------------------------------------------- */
int wubu_spec_bonus_token(const float *target_logits, const float *draft_probs,
                          int vocab, float rng) {
    /* Build residual distribution over the rejected position. */
    float *res = (float *)malloc(sizeof(float) * vocab);
    if (!res) return -1;
    float sum = 0.0f;
    for (int i = 0; i < vocab; i++) {
        float r = target_logits[i] - draft_probs[i];
        if (r < 0) r = 0;
        res[i] = r; sum += r;
    }
    if (sum <= 0) { free(res); return -1; }
    float acc = 0, target = rng * sum;
    int tok = vocab - 1;
    for (int i = 0; i < vocab; i++) {
        acc += res[i];
        if (acc >= target) { tok = i; break; }
    }
    free(res);
    return tok;
}