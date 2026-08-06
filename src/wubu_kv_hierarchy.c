/* wubu_kv_hierarchy.c — hyperbolic KV namespace hierarchy
 *
 * Maps /kv/in/<path> into the Poincaré disk so the directory tree
 * becomes a hyperbolic tree: shallow paths near center, deep paths
 * near the boundary, sibling files spread by angle.
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 6 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_kv_hierarchy.h"
#include "wubu_mobius.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

wubu_kv_hyperbolic_cfg_t wubu_kv_hyperbolic_default_cfg(void) {
    wubu_kv_hyperbolic_cfg_t cfg;
    cfg.R = 1.5f;
    cfg.origin[0] = 0.0f;
    cfg.origin[1] = 0.0f;
    cfg.dim = 2;
    return cfg;
}

/* Split a KV path into depth-components and compute the (depth, sibling)
 * for the leaf. The relative path under kv_root determines the tree
 * position. Each component at each level gets a sibling index based on
 * hash, so the angle spreads siblings around the circle. */
static void compute_path_position(const char *kv_root, const char *path,
                                   float *out_depth, float *out_frac) {
    /* Strip the kv_root prefix */
    size_t root_len = strlen(kv_root);
    const char *rel = path;
    if (strncmp(path, kv_root, root_len) == 0)
        rel = path + root_len;
    /* Skip leading slash */
    if (*rel == '/') rel++;

    /* Count path components (depth) and hash the leaf for sibling position */
    if (*rel == '\0') {
        /* The root itself */
        *out_depth = 0.0f;
        *out_frac = 0.0f;
        return;
    }

    /* Walk the components, counting depth */
    int depth = 0;
    const char *p = rel;
    while (*p) {
        if (*p == '/') depth++;
        p++;
    }

    /* The leaf is the last component. Hash it for sibling angle. */
    const char *last_slash = strrchr(rel, '/');
    const char *leaf = last_slash ? last_slash + 1 : rel;
    /* Simple FNV-1a hash for sibling distribution */
    unsigned int hash = 2166136261u;
    for (const char *c = leaf; *c; c++) {
        hash ^= (unsigned char)*c;
        hash *= 16777619u;
    }
    float frac = (float)(hash % 1000) / 1000.0f;

    *out_depth = (float)(depth + 1);
    *out_frac = frac;
}

wubu_kv_point_t wubu_kv_path_to_point(const char *kv_root,
                                       const char *path,
                                       const wubu_kv_hyperbolic_cfg_t *cfg) {
    wubu_kv_point_t pt;
    float R = cfg ? cfg->R : 1.5f;

    float depth, frac;
    compute_path_position(kv_root, path, &depth, &frac);

    /* radius = (1 - exp(-depth)) * R — asymptotic to R */
    float r = (1.0f - expf(-depth)) * R;
    /* Clamp to stay strictly inside the ball (artanh needs r < R) */
    if (r >= R * 0.999f) r = R * 0.999f;
    if (r < 0.0f) r = 0.0f;

    /* angle = 2π * frac */
    float theta = 2.0f * (float)M_PI * frac;

    pt.r = r;
    pt.theta = theta;
    pt.coords[0] = r * cosf(theta);
    pt.coords[1] = r * sinf(theta);

    return pt;
}

float wubu_kv_path_distance(const char *kv_root,
                             const char *path_a, const char *path_b,
                             const wubu_kv_hyperbolic_cfg_t *cfg) {
    wubu_kv_point_t pa = wubu_kv_path_to_point(kv_root, path_a, cfg);
    wubu_kv_point_t pb = wubu_kv_path_to_point(kv_root, path_b, cfg);
    float R = cfg ? cfg->R : 1.5f;
    /* Use the Möbius addition-based Poincaré distance from wubu_mobius */
    return wubu_poincare_dist(pa.coords, pb.coords, 2, R);
}

int wubu_kv_path_nearest(const char *kv_root,
                         const char **paths, int n_paths,
                         const char *query_path,
                         const wubu_kv_hyperbolic_cfg_t *cfg) {
    if (!paths || n_paths <= 0 || !query_path) return -1;
    int best = -1;
    float best_dist = INFINITY;
    for (int i = 0; i < n_paths; i++) {
        float d = wubu_kv_path_distance(kv_root, paths[i], query_path, cfg);
        if (d < best_dist) {
            best_dist = d;
            best = i;
        }
    }
    return best;
}

float wubu_kv_path_routing_score(const char *kv_root,
                                  const char *path_a, const char *path_b,
                                  const wubu_kv_hyperbolic_cfg_t *cfg) {
    float d = wubu_kv_path_distance(kv_root, path_a, path_b, cfg);
    float R = cfg ? cfg->R : 1.5f;
    /* score = exp(-d/R) — close files score near 1, far files → 0 */
    return expf(-d / R);
}
