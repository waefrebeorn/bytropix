/* wubu_kv_shrink.h — prune dead KV regions (Phase 10)
 *
 * After the model has attended to the KV namespace for some iterations,
 * some KV regions become "dead" — the model never attends to them
 * (attention mass ≈ 0 over multiple forwards). These waste memory.
 *
 * The shrink operator:
 *   1. Receives per-region utilization (from coherence stats: how often
 *      each /kv/in/<path> region was attended to).
 *   2. Regions below the utilization threshold for `cold_iters` consecutive
 *      forwards are marked for shrinking.
 *   3. Shrinking unmounts the KV region and returns its blocks to the
 *      freelist (wubu_kvfs), making them available for grow_kv.
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 10 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#ifndef WUBU_KV_SHRINK_H
#define WUBU_KV_SHRINK_H

#include <stddef.h>
#include "wubu_kvfs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_kv_shrink wubu_kv_shrink_t;

typedef struct {
    float util_threshold;   /* below this → cold (0.01) */
    int   cold_iters;       /* consecutive cold forwards to trigger shrink (10) */
    int   min_regions;      /* never shrink below this many regions (1) */
} wubu_kv_shrink_cfg_t;

wubu_kv_shrink_cfg_t wubu_kv_shrink_default_cfg(void);

/* Create the shrink operator over a filesystem. */
wubu_kv_shrink_t *wubu_kv_shrink_create(wubu_kvfs_t *fs,
                                         const wubu_kv_shrink_cfg_t *cfg);

/* Feed the per-region utilization from the latest forward pass.
 *
 * n_regions / paths / utilization: arrays of length n_regions.
 *   paths[i] = KV path (e.g. "/kv/in/foo.c")
 *   utilization[i] = fraction of total attention on this region [0, 1]
 * Returns 0 on success, -1 on error. */
int wubu_kv_shrink_feed(wubu_kv_shrink_t *s,
                         const char **paths, const float *utilization,
                         int n_regions);

/* Run the shrink sweep. Unmounts cold regions and returns the count
 * of regions pruned. The pruned paths are written to out_paths
 * (if non-NULL, caller must free each string + the array).
 *
 * Returns the number of regions pruned. */
int wubu_kv_shrink_sweep(wubu_kv_shrink_t *s, char ***out_paths);

/* Free the shrink operator. Does NOT free fs. */
void wubu_kv_shrink_free(wubu_kv_shrink_t *s);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_KV_SHRINK_H */
