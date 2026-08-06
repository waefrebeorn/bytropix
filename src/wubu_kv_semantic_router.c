/* wubu_kv_semantic_router.c — Poincaré routing bias for attention
 *
 * Translates the hyperbolic KV hierarchy into attention bias:
 * semantically/structurally related files (close in the Poincaré tree)
 * get a positive bias to the model's attention logits.
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 8 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_kv_semantic_router.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct wubu_kv_router {
    char                   root[256];
    const char            **paths;
    int                     n_paths;
    wubu_kv_hyperbolic_cfg_t cfg;
    float                   scale;
};

wubu_kv_router_t *wubu_kv_router_create(const char *root,
                                         const char **paths, int n_paths,
                                         const wubu_kv_hyperbolic_cfg_t *cfg,
                                         float scale) {
    if (!root || !paths || n_paths <= 0) return NULL;
    wubu_kv_router_t *rt = (wubu_kv_router_t *)calloc(1, sizeof(*rt));
    if (!rt) return NULL;
    strncpy(rt->root, root, sizeof(rt->root) - 1);
    rt->root[sizeof(rt->root) - 1] = '\0';
    /* Copy the path pointers (caller owns the strings) */
    rt->paths = (const char **)malloc((size_t)n_paths * sizeof(const char *));
    if (!rt->paths) { free(rt); return NULL; }
    for (int i = 0; i < n_paths; i++) {
        rt->paths[i] = paths[i];
    }
    rt->n_paths = n_paths;
    if (cfg) rt->cfg = *cfg;
    else rt->cfg = wubu_kv_hyperbolic_default_cfg();
    rt->scale = scale > 0 ? scale : 2.0f;
    return rt;
}

int wubu_kv_router_bias(const wubu_kv_router_t *rt,
                         const char *query_path,
                         float *out_bias) {
    if (!rt || !query_path || !out_bias) return -1;
    for (int i = 0; i < rt->n_paths; i++) {
        out_bias[i] = rt->scale * wubu_kv_path_routing_score(
            rt->root, query_path, rt->paths[i], &rt->cfg);
    }
    return 0;
}

float wubu_kv_router_bias_pair(const wubu_kv_router_t *rt,
                                const char *path_a,
                                const char *path_b) {
    if (!rt || !path_a || !path_b) return 0.0f;
    return rt->scale * wubu_kv_path_routing_score(
        rt->root, path_a, path_b, &rt->cfg);
}

int wubu_kv_router_paths(const wubu_kv_router_t *rt,
                          const char **out_paths) {
    if (!rt || !out_paths) return 0;
    for (int i = 0; i < rt->n_paths; i++)
        out_paths[i] = rt->paths[i];
    return rt->n_paths;
}

void wubu_kv_router_free(wubu_kv_router_t *rt) {
    if (!rt) return;
    free(rt->paths);
    free(rt);
}
