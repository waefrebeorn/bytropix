/*
 * wubu_amoeba.h -- THE AMOEBA: the diagnostic self-evolving model.
 *
 * The user: "a model that can evolve and get bigger and smaller like
 * an amoeba." The colony model (see docs/wubu-amoeba-design.md):
 *
 *   - the COLONY: a pool of specialized cells (experts), routed by
 *     the mixed agents (wubu_moe2)
 *   - the TISSUE: the hive (wubu_hive) -- dead cells are skip-marked,
 *     their slots recycled by the freelist (the amoeba's membrane)
 *   - the IMMUNE SYSTEM: per-cell diagnostics (utilization, gradient
 *     norm, loss delta, route entropy) -- this module
 *   - the OPERATORS: GROW (mitosis: split an overworked cell),
 *     SHRINK (apoptosis: prune a dead cell), STASIS (healthy band)
 *   - the FITNESS GATE: held-out loss + the Lean-verified prover +
 *     safety invariants; accepted variants archive, rejected ones
 *     roll back via the 5+1 recovery
 *
 * The Darwin Gödel Machine pattern (AGI_HOME_METAGAME H3): mutate ->
 * validate -> archive, empirically, open-ended, sandboxed. The model
 * can GROW from 35M to 70M and SHRINK back to 30M -- it adapts to the
 * task, not to a fixed config.
 *
 * Pure C11, opaque structs. The organs are self-contained modules;
 * this is the immune system that watches them.
 */
#ifndef WUBU_AMOEBA_H
#define WUBU_AMOEBA_H

#include <stdint.h>
#include <stddef.h>

#include "wubu_hive.h"
#include "wubu_moe2.h"
#include "wubu_prover.h"

/* per-cell diagnostic */
typedef struct {
    double utilization;    /* routes received / total routes */
    double grad_norm;      /* avg |dL/dW_e| over the batch */
    double loss_delta;     /* the held-out loss change if removed
                              (approx: -||g||^2/n, the quadratic) */
    double route_entropy;  /* the router's softmax entropy for the cell
                              (near-0 = always/never fires = signal) */
} wubu_amoeba_cell_t;

/* the colony vitals (the diagnosis output) */
typedef struct {
    int    n_cells;         /* live cells now */
    int    n_grow;          /* cells past the growth threshold */
    int    n_shrink;        /* cells past the death threshold */
    int    n_stasis;        /* cells in the healthy band */
    double total_live;      /* hive live count (the tissue) */
    double memory_used;     /* hive capacity */
    double fitness;         /* the current held-out loss */
    double prev_fitness;    /* the pre-mutation fitness */
    int    accepted;        /* mutations accepted (archived) */
    int    rejected;        /* mutations rolled back */
} wubu_amoeba_vitals_t;

/* the config (thresholds; the defaults follow the diagnostic rules) */
typedef struct {
    double grow_util;       /* utilization above this -> grow (0.7) */
    double grow_grad;       /* grad norm above this -> grow (relative) */
    double shrink_util;     /* utilization below this -> shrink (0.02) */
    double shrink_grad;     /* grad norm below this -> shrink (rel) */
    double entropy_min;     /* route entropy below this -> signal (0.05) */
    double loss_tol;        /* the mutation may raise loss by at most
                               this much and still be accepted (0.05) */
    double split_eps;       /* the mitosis perturbation (0.01) */
    int    max_cells;       /* the colony ceiling (safety) */
    int    min_cells;       /* the colony floor (safety) */
} wubu_amoeba_cfg_t;

/* the amoeba state */
typedef struct {
    wubu_amoeba_cfg_t cfg;
    wubu_hive_t      *tissue;     /* the hive (the membrane) */
    wubu_moe2_t      *agents;     /* the colony (the cells) */
    /* diagnostics */
    wubu_amoeba_cell_t *cells;    /* [max_cells] */
    int  n_cells;
    /* the mutation scratch (per split) */
    float *gate_parent;           /* the parent gate copy */
    /* the birth ledger: the tissue tokens in birth order (grow pushes,
     * shrink pops -- deterministic, reentrant, no statics) */
    uintptr_t *born;              /* [max_cells] */
    int  n_born;
    /* vitals + the archive ledger */
    wubu_amoeba_vitals_t vitals;
    uint32_t epoch;
} wubu_amoeba_t;

/* A1: init -- wire the organs; the caller owns them. */
int wubu_amoeba_init(wubu_amoeba_t *am, wubu_amoeba_cfg_t *cfg,
                     wubu_hive_t *tissue, wubu_moe2_t *agents);

/* A2: feed the per-cell gradient norms from the trainer (the immune
 * system's input). grad_norms[n_cells]. */
int wubu_amoeba_feed_grads(wubu_amoeba_t *am, const float *grad_norms);

/* A3: DIAGNOSE -- measure every cell. The route counts must already
 * be in the agents (wubu_moe2 tracks them); the grads come from A2. */
int wubu_amoeba_diagnose(wubu_amoeba_t *am);

/* A4: MUTATE -- grow/shrink/stasis per the diagnosis. Returns the
 * number of mutations applied. The 5+1 checkpoint is the caller's
 * (the recovery module owns it); this module only mutates. */
int wubu_amoeba_mutate(wubu_amoeba_t *am);

/* A5: VALIDATE -- the fitness gate. Returns 1 (accept) or 0 (reject).
 * The caller runs the held-out probe; this checks the loss tolerance
 * + the prover invariants (the Lean-verified theorems must still
 * hold after the mutation). */
int wubu_amoeba_validate(wubu_amoeba_t *am, double held_out_loss);

/* A6: COMMIT -- archive the accepted variant (append to the ledger)
 * or mark rejection (the caller does the 5+1 rollback). */
int wubu_amoeba_commit(wubu_amoeba_t *am, int accepted);

/* A7: the vitals (for the stats + the ledger). */
void wubu_amoeba_stats(const wubu_amoeba_t *am, char *buf, size_t cap);

/* A8: free (does NOT free the organs -- the caller owns them). */
void wubu_amoeba_free(wubu_amoeba_t *am);

#endif
