/* wubu_coherence.c -- PATHS #312 Verifiable Coherence: proof-gated
 * self-improvement. See wubu_coherence.h. */
#include "wubu_coherence.h"

int wubu_coherence_init(wubu_coherence_t *c) {
    if (!c) return -1;
    for (int i = 0; i < WUBU_COHERENCE_MAX_INVARIANTS; i++) {
        c->inv[i].fn = NULL;
        c->inv[i].name = NULL;
    }
    c->n_inv = 0;
    c->n_checked = 0;
    c->n_admitted = 0;
    c->n_rejected = 0;
    return 0;
}

int wubu_coherence_add(wubu_coherence_t *c, wubu_invariant_fn fn, const char *name) {
    if (!c || !fn) return -1;
    if (c->n_inv >= WUBU_COHERENCE_MAX_INVARIANTS) return -1;
    c->inv[c->n_inv].fn = fn;
    c->inv[c->n_inv].name = name;
    c->n_inv++;
    return 0;
}

static wubu_coherence_result_t prove(wubu_coherence_t *c,
        const void *current_state,
        wubu_state_transform_fn transform, void *transform_ctx,
        void *proposed_state, void *invariant_ctx) {
    wubu_coherence_result_t r = { 0, 0, 0, -1, NULL };
    /* Fail-closed: no invariants registered = cannot prove coherence. */
    if (!c || c->n_inv == 0 || !transform || !current_state || !proposed_state) {
        r.coherent = 0;
        c->n_checked++;
        c->n_rejected++;
        r.first_fail = -1;
        r.fail_name = "fail-closed: no proof available";
        return r;
    }
    /* Compute the proposed state via the transform. */
    if (transform(current_state, proposed_state, transform_ctx) != 0) {
        c->n_checked++;
        c->n_rejected++;
        r.fail_name = "transform failed";
        return r;
    }
    /* (1) baseline: every invariant holds on the current state. */
    for (int i = 0; i < c->n_inv; i++) {
        if (!c->inv[i].fn(current_state, invariant_ctx)) {
            r.baseline_ok = 0;
            r.first_fail = i;
            r.fail_name = c->inv[i].name;
            c->n_checked++; c->n_rejected++;
            return r;
        }
    }
    r.baseline_ok = 1;
    /* (2) proposed: every invariant holds on the proposed state. */
    for (int i = 0; i < c->n_inv; i++) {
        if (!c->inv[i].fn(proposed_state, invariant_ctx)) {
            r.proposed_ok = 0;
            r.first_fail = i;
            r.fail_name = c->inv[i].name;
            c->n_checked++; c->n_rejected++;
            return r;
        }
    }
    r.proposed_ok = 1;
    r.coherent = 1;
    c->n_checked++; c->n_admitted++;
    return r;
}

wubu_coherence_result_t wubu_coherence_prove(wubu_coherence_t *c,
        const void *current_state,
        wubu_state_transform_fn transform, void *transform_ctx,
        void *proposed_state, void *invariant_ctx) {
    return prove(c, current_state, transform, transform_ctx, proposed_state, invariant_ctx);
}

int wubu_coherence_admit(wubu_coherence_t *c,
        const void *current_state,
        wubu_state_transform_fn transform, void *transform_ctx,
        void *proposed_state, void *invariant_ctx) {
    return prove(c, current_state, transform, transform_ctx, proposed_state, invariant_ctx).coherent;
}
