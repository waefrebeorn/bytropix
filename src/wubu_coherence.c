/* wubu_coherence.c -- PATHS #312 Verifiable Coherence: proof-gated
 * self-improvement. See wubu_coherence.h. */
#include "wubu_coherence.h"
#include <limits.h>
#include <stddef.h>

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

/* DA-1: screen a conjunction of comparisons over ONE integer field before any
 * evaluation is spent. The feasible set is an interval [lo, hi]; every
 * declaration can only narrow it, so the set is empty exactly when lo > hi, or
 * when it is the single point {lo} and some NE excludes it. */
int wubu_da1_screen(const wubu_inv_decl_t *decls, int n, const char **conflict)
{
    if (conflict) *conflict = NULL;
    if (!decls || n <= 0) return 0;

    long lo = LONG_MIN, hi = LONG_MAX;
    long ne[8];
    const char *ne_name[8];
    int n_ne = 0;
    const char *why = NULL;

    for (int i = 0; i < n; i++) {
        long v = decls[i].value;
        switch (decls[i].op) {
        case WUBU_CMP_LE:
            if (v < hi) hi = v;
            if (lo > hi) why = decls[i].name;
            break;
        case WUBU_CMP_GE:
            if (v > lo) lo = v;
            if (lo > hi) why = decls[i].name;
            break;
        case WUBU_CMP_EQ:
            if (v > lo) lo = v;
            if (v < hi) hi = v;
            if (lo > hi) why = decls[i].name;
            break;
        case WUBU_CMP_NE:
            if (n_ne < 8) { ne[n_ne] = v; ne_name[n_ne] = decls[i].name; n_ne++; }
            break;
        default:
            if (conflict) *conflict = decls[i].name;
            return 0;
        }
        if (why) break;
    }

    if (!why && lo == hi) {
        for (int i = 0; i < n_ne; i++) {
            if (ne[i] == lo) { why = ne_name[i]; break; }
        }
    }

    if (why) {
        if (conflict) *conflict = why;
        return 0;
    }
    return 1;
}
