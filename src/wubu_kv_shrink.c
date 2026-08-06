/* wubu_kv_shrink.c — prune dead KV regions
 *
 * Tracks per-region utilization over consecutive forward passes.
 * Regions that fall below the threshold for `cold_iters` consecutive
 * passes are unmounted, returning blocks to the freelist.
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 10 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_kv_shrink.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_REGIONS 256

typedef struct {
    char  path[128];
    int   cold_streak;  /* consecutive cold iterations */
    int   active;       /* 1 = tracked, 0 = pruned */
} shrink_region_t;

struct wubu_kv_shrink {
    wubu_kvfs_t          *fs;
    wubu_kv_shrink_cfg_t  cfg;
    shrink_region_t       regions[MAX_REGIONS];
    int                   n_regions;
};

wubu_kv_shrink_cfg_t wubu_kv_shrink_default_cfg(void) {
    wubu_kv_shrink_cfg_t cfg;
    cfg.util_threshold = 0.01f;
    cfg.cold_iters = 10;
    cfg.min_regions = 1;
    return cfg;
}

wubu_kv_shrink_t *wubu_kv_shrink_create(wubu_kvfs_t *fs,
                                         const wubu_kv_shrink_cfg_t *cfg) {
    if (!fs || !cfg) return NULL;
    wubu_kv_shrink_t *s = (wubu_kv_shrink_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->fs = fs;
    s->cfg = *cfg;
    s->n_regions = 0;
    return s;
}

void wubu_kv_shrink_free(wubu_kv_shrink_t *s) {
    if (!s) return;
    free(s);
}

/* Find or add a region by path */
static int region_find_or_add(wubu_kv_shrink_t *s, const char *path) {
    for (int i = 0; i < s->n_regions; i++) {
        if (s->regions[i].active &&
            strcmp(s->regions[i].path, path) == 0)
            return i;
    }
    if (s->n_regions >= MAX_REGIONS) return -1;
    int idx = s->n_regions++;
    strncpy(s->regions[idx].path, path, sizeof(s->regions[idx].path) - 1);
    s->regions[idx].path[sizeof(s->regions[idx].path) - 1] = '\0';
    s->regions[idx].cold_streak = 0;
    s->regions[idx].active = 1;
    return idx;
}

int wubu_kv_shrink_feed(wubu_kv_shrink_t *s,
                         const char **paths, const float *utilization,
                         int n_regions) {
    if (!s || !paths || !utilization || n_regions <= 0) return -1;
    for (int i = 0; i < n_regions; i++) {
        int idx = region_find_or_add(s, paths[i]);
        if (idx < 0) continue;
        if (utilization[i] < s->cfg.util_threshold) {
            s->regions[idx].cold_streak++;
        } else {
            s->regions[idx].cold_streak = 0;
        }
    }
    return 0;
}

int wubu_kv_shrink_sweep(wubu_kv_shrink_t *s, char ***out_paths) {
    if (!s) return 0;
    int pruned = 0;
    /* Count to-prune regions */
    int n_prune = 0;
    for (int i = 0; i < s->n_regions; i++) {
        if (s->regions[i].active &&
            s->regions[i].cold_streak >= s->cfg.cold_iters) {
            /* Don't shrink below min_regions */
            if (s->n_regions - pruned <= s->cfg.min_regions) break;
            n_prune++;
        }
    }
    if (n_prune == 0) return 0;

    char **paths = (char **)malloc((size_t)n_prune * sizeof(char *));
    if (!paths) return 0;

    for (int i = 0; i < s->n_regions && pruned < n_prune; i++) {
        if (!s->regions[i].active) continue;
        if (s->regions[i].cold_streak < s->cfg.cold_iters) continue;
        /* Check min_regions */
        if (s->n_regions - pruned <= s->cfg.min_regions) break;
        /* Unmount the KV region */
        wubu_kvfs_unmount(s->fs, s->regions[i].path);
        s->regions[i].active = 0;
        paths[pruned] = strdup(s->regions[i].path);
        if (!paths[pruned]) paths[pruned] = (char *)s->regions[i].path;
        pruned++;
    }

    if (out_paths) {
        *out_paths = paths;
    } else {
        /* Free paths we allocated */
        for (int i = 0; i < pruned; i++)
            if (paths[i] != s->regions[i].path) free(paths[i]);
        free(paths);
    }
    return pruned;
}
