/*
 * wubu_spec_cascade.c — Cascade speculative decoding (Area K).
 * Pure C11, self-contained. Two cascade flavors:
 *   1. N-gram cascade: prompt n-gram statistics as drafter (zero extra model)
 *   2. Self-cascade: small local model as drafter (e.g., A1-4B for Qwen-27B)
 *
 * Self-cascade implementation is in wubu_self_cascade.c
 * N-gram cascade implementation is in wubu_ngram_cascade.c
 */
#include "wubu_spec_cascade.h"
#include "wubu_spec_decode.h"
#include "wubu_ngram.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---------- N-gram cascade ---------- */

wubu_ngram_cascade_t *wubu_ngram_cascade_create(const int *ctx, int ctx_len, int order,
                                                 int draft_depth, int defer_threshold) {
    (void)defer_threshold;
    wubu_ngram_cascade_t *c = (wubu_ngram_cascade_t *)malloc(sizeof(*c));
    if (!c) return NULL;
    c->ngram = wubu_ngram_create(ctx, ctx_len, order);
    c->draft_depth = draft_depth > 0 ? draft_depth : 4;
    return c;
}

void wubu_ngram_cascade_free(wubu_ngram_cascade_t *c) {
    if (c) {
        if (c->ngram) wubu_ngram_free(c->ngram);
        free(c);
    }
}

int wubu_ngram_cascade_propose(wubu_ngram_cascade_t *c, int *out_tokens, float *out_probs) {
    if (!c || !c->ngram) return 0;
    int proposed = wubu_ngram_propose(c->ngram, c->draft_depth, out_tokens);
    if (proposed > 0 && out_probs) {
        for (int i = 0; i < proposed; i++) out_probs[i] = 1.0f / proposed;
    }
    return proposed;
}

void wubu_ngram_cascade_update(wubu_ngram_cascade_t *c, const int *accepted, int n_accepted) {
    if (!c || !c->ngram) return;
    wubu_ngram_update_context(c->ngram, accepted, n_accepted);
}

/* ---------- Cascade verification with deferral ---------- */

int wubu_cascade_verify(const int *candidates, const int *parent,
                        const float *draft_probs, const float *target_logits,
                        int n_cand, int vocab, int *accepted, int max_accepted,
                        float rng, float defer_eps) {
    (void)vocab;
    int accepted_n = 0;

    for (int i = 0; i < n_cand && accepted_n < max_accepted; i++) {
        int tok = candidates[i];
        int p = parent[i];
        int parent_accepted = (p < 0) ? 1 : (p < accepted_n);
        if (!parent_accepted) continue;

        float p_target = target_logits[tok];
        float p_draft  = draft_probs[i] > 1e-9f ? draft_probs[i] : 1e-9f;

        if (p_draft > p_target * (1.0f + defer_eps)) {
            accepted[accepted_n++] = tok;
            continue;
        }

        if (p_target >= p_draft) {
            accepted[accepted_n++] = tok;
        } else if (rng < p_target / p_draft) {
            accepted[accepted_n++] = tok;
        } else {
            break;
        }
    }
    return accepted_n;
}

/* ---------- High-level cascade step ---------- */

int wubu_cascade_step_ngram(wubu_ngram_cascade_t *drafter,
                            const int *context, int ctx_len,
                            const float *target_logits,
                            int *out_accepted, int max_accepted,
                            float rng, float defer_eps) {
    int draft_tokens[16];
    float draft_probs[16];

    int n_draft = wubu_ngram_cascade_propose(drafter, draft_tokens, draft_probs);
    if (n_draft == 0) return 0;

    int candidates[16], parent[16];
    for (int i = 0; i < n_draft; i++) {
        candidates[i] = draft_tokens[i];
        parent[i] = (i == 0) ? -1 : (i - 1);
    }

    int n_accepted = wubu_cascade_verify(candidates, parent, draft_probs,
                                         target_logits, n_draft,
                                         0, out_accepted, max_accepted,
                                         rng, defer_eps);

    if (n_accepted > 0) {
        wubu_ngram_cascade_update(drafter, out_accepted, n_accepted);
    }
    return n_accepted;
}

int wubu_cascade_step_self(wubu_self_cascade_t *drafter,
                           const int *context, int ctx_len,
                           const float *target_logits,
                           int *out_accepted, int max_accepted,
                           float rng, float defer_eps) {
    int draft_tokens[16];
    float draft_probs[16];

    int n_draft = wubu_self_cascade_propose(drafter, context, ctx_len,
                                            draft_tokens, draft_probs);
    if (n_draft == 0) return 0;

    int candidates[16], parent[16];
    for (int i = 0; i < n_draft; i++) {
        candidates[i] = draft_tokens[i];
        parent[i] = (i == 0) ? -1 : (i - 1);
    }

    int n_accepted = wubu_cascade_verify(candidates, parent, draft_probs,
                                         target_logits, n_draft,
                                         drafter->vocab, out_accepted, max_accepted,
                                         rng, defer_eps);
    return n_accepted;
}