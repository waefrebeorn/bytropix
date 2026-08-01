/*
 * wubu_early_exit.c — Early-exit + self-speculative verify (doc J03).
 * See header. Self-contained C11.
 */
#include "wubu_early_exit.h"
#include <stdlib.h>
#include <math.h>

struct wubu_early_exit {
    float exit_threshold;   /* hidden delta below which we may exit early */
    int   max_draft;        /* self-spec draft depth */
    int   n_early_exits;    /* cumulative count */
    int   n_spec_accepts;   /* cumulative accepted draft tokens */
};

wubu_early_exit_t *wubu_early_exit_create(float exit_threshold, int max_draft) {
    wubu_early_exit_t *e = (wubu_early_exit_t *)calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->exit_threshold = (exit_threshold > 0.0f) ? exit_threshold : 1e30f;
    e->max_draft = max_draft > 0 ? max_draft : 0;
    return e;
}

void wubu_early_exit_free(wubu_early_exit_t *e) { free(e); }

int wubu_early_exit_should_stop(const wubu_early_exit_t *e,
                                int layer, int n_layers,
                                float hidden_delta, float hidden_norm) {
    if (!e) return 0;
    /* Never exit in the first 25% of layers (need a stable representation) or
     * on the final layer (must produce logits). */
    if (layer < n_layers / 4) return 0;
    if (layer >= n_layers - 1) return 0;
    /* Relative convergence: delta small compared to state magnitude. */
    float rel = (hidden_norm > 1e-6f) ? hidden_delta / hidden_norm : hidden_delta;
    if (rel < e->exit_threshold) {
        /* Only count if gate is actually enabled (finite threshold). */
        if (e->exit_threshold < 1e29f) return 1;
    }
    return 0;
}

int wubu_early_exit_draft(int depth,
                          const int *shallow_top1, const int *full_top1,
                          int *out_draft) {
    if (depth <= 0 || !shallow_top1 || !full_top1 || !out_draft) return 0;
    int n = 0;
    for (int i = 0; i < depth; i++) {
        /* Draft token i is safe only if shallow trunk agrees with full model at
         * this step. This is the self-speculative consistency check. */
        if (shallow_top1[i] == full_top1[i]) {
            out_draft[n++] = shallow_top1[i];
        } else {
            /* Disagreement => stop drafting; verify up to here. */
            break;
        }
    }
    return n;
}

int wubu_early_exit_verify(const wubu_early_exit_t *e,
                           const int *draft, const float *draft_probs,
                           int n, float threshold, int *accepted) {
    if (!e || !draft || !draft_probs || !accepted) return 0;
    int acc = 0;
    for (int i = 0; i < n; i++) {
        if (draft_probs[i] >= threshold) {
            acc++;
            /* count for stats if gate enabled */
            if (e->exit_threshold < 1e29f || e->max_draft > 0)
                ((wubu_early_exit_t *)e)->n_spec_accepts++;
        } else {
            break; /* first rejection ends the accepted run */
        }
    }
    *accepted = acc;
    return (acc == n) ? 1 : 0;
}

void wubu_early_exit_stats(const wubu_early_exit_t *e,
                           int *early_exits, int *spec_accepts) {
    if (early_exits)  *early_exits  = e ? e->n_early_exits  : 0;
    if (spec_accepts) *spec_accepts = e ? e->n_spec_accepts : 0;
}
