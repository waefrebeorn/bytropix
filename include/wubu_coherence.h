#ifndef WUBU_COHERENCE_H
#define WUBU_COHERENCE_H
#include <stddef.h>

/* wubu_coherence.h -- PATHS #312 Verifiable Coherence: proof-gated
 * self-improvement.
 *
 * A self-improvement mutation is admitted ONLY if it preserves the system's
 * non-negotiable invariants. The gate holds a set of invariants, each a
 * predicate over a generic state pointer (the current world state). To
 * "prove" a proposed mutation coherent, the caller supplies a transform
 * function that computes the PROPOSED state; the gate evaluates every
 * invariant against BOTH the current and proposed states and requires:
 *   (1) every invariant holds on the current state (baseline is coherent), AND
 *   (2) every invariant holds on the proposed state (mutation preserves them).
 * If either fails, the mutation is REJECTED (fail-closed). This is the
 * runtime enforcement of "a mutation needs a consistency proof".
 *
 * Deterministic, no allocation on the hot path, fail-closed: an empty
 * invariant set or a NULL transform is treated as NOT coherent (you cannot
 * prove coherence you did not check). Real work (no stubs). */

#define WUBU_COHERENCE_MAX_INVARIANTS 32
#define WUBU_COHERENCE_NAMELEN 48

typedef int (*wubu_invariant_fn)(const void *state, void *ctx);
typedef int (*wubu_state_transform_fn)(const void *from, void *to, void *ctx);

typedef struct {
    wubu_invariant_fn fn;
    const char       *name;
} wubu_invariant_t;

typedef struct {
    wubu_invariant_t inv[WUBU_COHERENCE_MAX_INVARIANTS];
    int n_inv;
    /* observability */
    int n_checked;   /* total mutations gated */
    int n_admitted;  /* mutations that passed the coherence proof */
    int n_rejected;  /* mutations rejected */
} wubu_coherence_t;

/* Result of a coherence proof. */
typedef struct {
    int coherent;        /* 1 = mutation admitted, 0 = rejected */
    int baseline_ok;     /* 1 = all invariants held on current state */
    int proposed_ok;     /* 1 = all invariants held on proposed state */
    int first_fail;      /* index of first failing invariant, or -1 */
    const char *fail_name; /* name of first failing invariant, or NULL */
} wubu_coherence_result_t;

int  wubu_coherence_init(wubu_coherence_t *c);
/* Register an invariant predicate. Returns 0/-1. */
int  wubu_coherence_add(wubu_coherence_t *c, wubu_invariant_fn fn, const char *name);
/* Prove a mutation coherent: evaluate all invariants on current + proposed
 * state (produced by transform). Returns the proof result. Fail-closed. */
wubu_coherence_result_t wubu_coherence_prove(wubu_coherence_t *c,
        const void *current_state,
        wubu_state_transform_fn transform, void *transform_ctx,
        void *proposed_state, void *invariant_ctx);
/* Convenience: 1 = admit the mutation (coherent), 0 = reject. */
int  wubu_coherence_admit(wubu_coherence_t *c,
        const void *current_state,
        wubu_state_transform_fn transform, void *transform_ctx,
        void *proposed_state, void *invariant_ctx);

#endif
