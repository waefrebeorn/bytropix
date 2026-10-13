#ifndef WUBU_HYPMOE_H
#define WUBU_HYPMOE_H

/* wubu_hypmoe.h -- BRAIN #116 hyperbolic MoE routing.
 *
 * Hyperbolic Mixture-of-Experts routing: experts and token embeddings live in
 * the Poincare disk; a token is routed to the experts it is CLOSEST to in
 * HYPERBOLIC distance (not Euclidean), which better captures hierarchical
 * expert specialization (some experts are "broad", others "narrow/deep" in the
 * tree). This module embeds token + expert vectors into the disk and routes
 * each token to its top-k nearest experts by hyperbolic distance. Deterministic,
 * real work (no stubs). */

#define WUBU_HM_DIM 8
#define WUBU_HM_MAX_EXPERTS 16

typedef struct {
    double center[WUBU_HM_DIM];   /* expert centroid in the disk */
    int id;
} wubu_hmoe_expert_t;

typedef struct {
    wubu_hmoe_expert_t experts[WUBU_HM_MAX_EXPERTS];
    int n_experts;
    double curvature;             /* negative, e.g. -1 */
} wubu_hmoe_t;

/* Initialise with a curvature. */
int wubu_hmoe_init(wubu_hmoe_t *m, double curvature);

/* Register an expert at a centroid. Returns 0/-1. */
int wubu_hmoe_add_expert(wubu_hmoe_t *m, int id, const double *center);

/* Route a token embedding to its top-k nearest experts (by hyperbolic
 * distance). Returns k (or fewer if not enough experts); fills expert_ids[]
 * with the nearest-first ids and dists[] with their distances. */
int wubu_hmoe_route(const wubu_hmoe_t *m, const double *token, int k,
                    int *expert_ids, double *dists);

#endif
