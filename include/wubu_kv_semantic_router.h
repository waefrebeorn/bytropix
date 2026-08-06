/* wubu_kv_semantic_router.h — Poincaré routing bias for attention (Phase 8)
 *
 * The KV namespace has a hyperbolic hierarchy (wubu_kv_hierarchy).
 * This module translates that hierarchy into an attention BIAS:
 * when the model's query token attends over context tokens from
 * /kv/in/, the attention is biased by the Poincaré distance between
 * the query's source file and each context file.
 *
 *   bias[i] = scale * routing_score(query_file, context_file[i])
 *           = scale * exp(-dist / R)
 *
 * Files that are structurally related (e.g., src/foo.c and src/foo_test.c)
 * are close in the Poincaré tree and get a positive attention bias.
 * Unrelated files (src/foo.c and docs/readme.md) get ~0 bias.
 *
 * This is the "semantic routing" that makes the KV-cache-as-filesystem
 * work at inference time: the model doesn't need to learn which files
 * are related — the hyperbolic tree encodes it, and we add it as a
 * bias to the attention logits before softmax.
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 8 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_KV_SEMANTIC_ROUTER_H
#define WUBU_KV_SEMANTIC_ROUTER_H

#include <stddef.h>
#include "wubu_kv_hierarchy.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct wubu_kv_router wubu_kv_router_t;

/* Create the router over a set of KV paths with a shared root.
 *
 * root:  the KV namespace root (e.g., "/kv/in")
 * paths: array of KV paths (full paths like /kv/in/src/foo.c)
 * n_paths: number of paths
 * cfg:   hyperbolic config (pass NULL for defaults)
 * scale: bias scale factor (default 2.0 — strong enough to influence
 *        softmax but not dominate raw attention logits)
 * Returns NULL on failure. */
wubu_kv_router_t *wubu_kv_router_create(const char *root,
                                         const char **paths, int n_paths,
                                         const wubu_kv_hyperbolic_cfg_t *cfg,
                                         float scale);

/* Compute the attention bias vector for a query file.
 *
 * query_path: the KV path the query tokens come from
 * out_bias:   [n_paths] — bias to add to attention logits
 *             (higher = more attend to that file)
 *             Bias = scale * routing_score(query_path, paths[i])
 *
 * Returns 0 on success, -1 on error (unknown path, etc.). */
int wubu_kv_router_bias(const wubu_kv_router_t *rt,
                         const char *query_path,
                         float *out_bias);

/* Compute the bias between two specific paths.
 * Returns routing_score(a, b) = exp(-dist(a,b) / R), scaled. */
float wubu_kv_router_bias_pair(const wubu_kv_router_t *rt,
                                const char *path_a,
                                const char *path_b);

/* Get the router's path list (for diagnostics).
 * Returns n_paths; out_paths is filled with pointers (not owned). */
int wubu_kv_router_paths(const wubu_kv_router_t *rt,
                          const char **out_paths);

/* Free the router. */
void wubu_kv_router_free(wubu_kv_router_t *rt);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KV_SEMANTIC_ROUTER_H */
