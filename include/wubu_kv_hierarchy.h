/* wubu_kv_hierarchy.h — hyperbolic KV namespace hierarchy (Phase 6)
 *
 * Maps the /kv/in/<path> namespace into the Poincaré ball so that
 * semantically/structurally related files are close in hyperbolic space.
 * The Poincaré distance then gives a natural routing metric:
 * the model attends to nearby files in the hierarchy.
 *
 * Path structure → Poincaré embedding:
 *   The path hierarchy IS the hyperbolic tree. Each path component
 *   (directory or file) is a node. The depth determines the radius
 *   (deeper = further from the center), and the sibling index determines
 *   the angle. This is the canonical hyperbolic tree embedding.
 *
 *   - root (/kv/in/) → origin (0, 0) — the center mass
 *   - depth d → radius r = (1 - e^(-d)) * R_ball
 *     (approaches R_ball as depth → ∞, staying in the ball)
 *   - sibling k out of K → angle θ = 2πk / K
 *
 * The grow operator (wubu_grow_kv) uses this to find the nearest
 * neighbor file in hyperbolic space and place new KV blocks nearby.
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 6 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_KV_HIERARCHY_H
#define WUBU_KV_HIERARCHY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The Poincaré ball configuration */
typedef struct {
    float R;          /* ball radius (curvature c = 1/R²) */
    float origin[2];  /* center (x, y) — typically {0, 0} */
    int   dim;        /* embedding dimension (2 for planar tree) */
} wubu_kv_hyperbolic_cfg_t;

/* Default config: R=1.5 (matches R_POINCARE_COARSE from wubu_moe_hyperbolic) */
wubu_kv_hyperbolic_cfg_t wubu_kv_hyperbolic_default_cfg(void);

/* A point in the Poincaré ball */
typedef struct {
    float coords[2];  /* (x, y) on the Poincaré disk */
    float r;          /* polar radius (0 ≤ r < R) */
    float theta;      /* polar angle */
} wubu_kv_point_t;

/* Map a KV path to a point in the Poincaré ball.
 *
 * The path hierarchy maps to hyperbolic depth:
 *   /kv/in/            → depth 0, center
 *   /kv/in/src/        → depth 1
 *   /kv/in/src/foo.c   → depth 2
 *   /kv/in/src/foo.c/grow0 → depth 3
 *
 * radius = (1 - exp(-depth)) * R  (approaches R asymptotically)
 * angle  = 2π * (sibling_index / num_siblings)
 *
 * kv_root: the base path inside the KV namespace (e.g., "/kv/in")
 * path:    the full KV path (e.g., "/kv/in/src/foo.c")
 * Returns the Poincaré point. */
wubu_kv_point_t wubu_kv_path_to_point(const char *kv_root,
                                       const char *path,
                                       const wubu_kv_hyperbolic_cfg_t *cfg);

/* Compute the Poincaré (geodesic) distance between two KV paths.
 * This is the hyperbolic distance between their points on the ball.
 * Returns d(x, y) = R * artanh(||(-x) ⊕ y|| / R). */
float wubu_kv_path_distance(const char *kv_root,
                             const char *path_a, const char *path_b,
                             const wubu_kv_hyperbolic_cfg_t *cfg);

/* Find the nearest neighbor of a query path among a list of KV paths,
 * by Poincaré distance. Returns the index of the nearest path, or -1.
 *
 * paths: array of KV paths (null-terminated strings)
 * n_paths: number of paths
 * query_path: the path to find the NN for */
int wubu_kv_path_nearest(const char *kv_root,
                         const char **paths, int n_paths,
                         const char *query_path,
                         const wubu_kv_hyperbolic_cfg_t *cfg);

/* Compute a routing score between two KV paths: higher = closer in the
 * hierarchy. score = exp(-distance / R) ∈ (0, 1].
 * This score can be used as an attention bias in the model's routing. */
float wubu_kv_path_routing_score(const char *kv_root,
                                  const char *path_a, const char *path_b,
                                  const wubu_kv_hyperbolic_cfg_t *cfg);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KV_HIERARCHY_H */
