/*
 * wubu_self_cascade.c — Self-cascade drafter (small local model).
 * Pure C11. Calls a provided small-model forward function.
 */
#include "wubu_spec_cascade.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct wubu_self_cascade_t {
    void *model_ctx;
    int (*forward)(void *model_ctx, const int *tokens, int n_tokens, float *logits, int vocab);
    int vocab;
    int draft_depth;
};

wubu_self_cascade_t *wubu_self_cascade_create(void *model_ctx,
                                               int (*forward)(void *, const int *, int, float *, int),
                                               int vocab, int draft_depth,
                                               int defer_threshold) {
    (void)defer_threshold;
    wubu_self_cascade_t *c = (wubu_self_cascade_t *)malloc(sizeof(*c));
    if (!c) return NULL;
    c->model_ctx = model_ctx;
    c->forward = forward;
    c->vocab = vocab;
    c->draft_depth = draft_depth > 0 ? draft_depth : 4;
    return c;
}

void wubu_self_cascade_free(wubu_self_cascade_t *c) {
    if (c) free(c);
}

int wubu_self_cascade_propose(wubu_self_cascade_t *c,
                               const int *context, int ctx_len,
                               int *out_tokens, float *out_probs) {
    if (!c || !c->forward) return 0;

    int *tokens = (int *)malloc(sizeof(int) * (ctx_len + c->draft_depth));
    if (!tokens) return 0;
    memcpy(tokens, context, sizeof(int) * ctx_len);

    int proposed = 0;
    float *logits = (float *)malloc(sizeof(float) * c->vocab);
    if (!logits) { free(tokens); return 0; }

    for (int step = 0; step < c->draft_depth; step++) {
        int rc = c->forward(c->model_ctx, tokens, ctx_len + step, logits, c->vocab);
        if (rc != 0) break;

        int best = 0; float best_val = logits[0];
        for (int v = 1; v < c->vocab; v++) {
            if (logits[v] > best_val) { best_val = logits[v]; best = v; }
        }

        tokens[ctx_len + step] = best;
        out_tokens[proposed] = best;
        if (out_probs) {
            float max_logit = logits[0];
            for (int v = 1; v < c->vocab; v++) if (logits[v] > max_logit) max_logit = logits[v];
            float sum = 0;
            for (int v = 0; v < c->vocab; v++) sum += expf(logits[v] - max_logit);
            out_probs[proposed] = expf(logits[best] - max_logit) / sum;
        }
        proposed++;
    }

    free(tokens);
    free(logits);
    return proposed;
}