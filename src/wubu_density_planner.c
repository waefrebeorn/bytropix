/* wubu_density_planner.c — information density planning (AN23-core)
 *
 * The AGI's resource allocator. Decides what data gets absorbed into
 * model weights (permanent memory) vs stays in KV cache (context
 * memory) vs gets pruned entirely.
 *
 * density = coherence.score / n_tokens (bits of understanding per token)
 *
 * High-density files (high coherence, low weight cost) → ABSORB
 * Mid-density files → KEEP in KV cache at reduced precision
 * Low-density files → PRUNE (shrink)
 *
 * Design: docs/wubu1-scalable-model-design.md §Density Planning.
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_density_planner.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_DENSITY_FILES 256

struct wubu_density_planner {
    wubu_kv_embedding_t  *kv;
    wubu_scalable_model_t *model;
    wubu_kv_tiering_t     *tiering;
    wubu_kv_shrink_t      *shrink;
    wubu_density_planner_cfg_t cfg;
    wubu_absorb_record_t   records[MAX_DENSITY_FILES];
    int                    n_records;
    float                   densities[MAX_DENSITY_FILES];
    int                     indices[MAX_DENSITY_FILES];
    size_t absorbed_bytes;
    size_t kv_bytes;
    size_t pruned_bytes;
};

wubu_density_planner_cfg_t wubu_density_planner_default_cfg(void) {
    wubu_density_planner_cfg_t cfg;
    cfg.absorb_threshold = 0.001f;
    cfg.keep_threshold = 0.0001f;
    cfg.prune_threshold = 0.00005f;
    cfg.weight_budget = 64 * 1024 * 1024;
    cfg.kv_budget = 128 * 1024 * 1024;
    cfg.absorb_batch = 8;
    return cfg;
}

wubu_density_planner_t *wubu_density_planner_create(
    wubu_kv_embedding_t *kv,
    wubu_scalable_model_t *model,
    wubu_kv_tiering_t *tiering,
    wubu_kv_shrink_t *shrink,
    const wubu_density_planner_cfg_t *cfg) {
    if (!kv || !cfg) return NULL;
    wubu_density_planner_t *p = (wubu_density_planner_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->kv = kv;
    p->model = model;
    p->tiering = tiering;
    p->shrink = shrink;
    p->cfg = *cfg;
    p->n_records = 0;
    p->absorbed_bytes = p->kv_bytes = p->pruned_bytes = 0;
    return p;
}

void wubu_density_planner_free(wubu_density_planner_t *p) {
    if (!p) return;
    free(p);
}

/* Simple insertion sort by density descending */
static void sort_indices_by_density_desc(int *indices, const float *densities, int n) {
    for (int i = 1; i < n; i++) {
        int key_idx = indices[i];
        float key_den = densities[key_idx];
        int j = i - 1;
        while (j >= 0 && densities[indices[j]] < key_den) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = key_idx;
    }
}

int wubu_density_planner_cycle(wubu_density_planner_t *planner,
                                const char **paths,
                                const wubu_coherence_t *coherence,
                                int n_files,
                                wubu_absorb_record_t **out_absorbed,
                                int *out_n_absorbed) {
    if (!planner || !paths || !coherence || n_files <= 0) return -1;
    if (n_files > MAX_DENSITY_FILES) n_files = MAX_DENSITY_FILES;

    /* Step 1: compute density for each file */
    for (int i = 0; i < n_files; i++) {
        size_t n_tokens = (coherence[i].n_tokens > 0)
                         ? (size_t)coherence[i].n_tokens : 1;
        planner->densities[i] = coherence[i].score / (float)n_tokens;
        planner->indices[i] = i;
    }

    /* Step 2: sort by density descending (insertion sort) */
    sort_indices_by_density_desc(planner->indices, planner->densities, n_files);

    /* Step 3: reset stats */
    planner->n_records = 0;
    planner->absorbed_bytes = 0;
    planner->kv_bytes = 0;
    planner->pruned_bytes = 0;

    /* Step 4: classify each file into absorb / keep / prune bands */
    int n_absorbed = 0;
    size_t weight_used = 0;

    for (int i = 0; i < n_files; i++) {
        int idx = planner->indices[i];
        float density = planner->densities[idx];
        const char *path = paths[idx];
        float score = coherence[idx].score;
        size_t n_tokens = (coherence[idx].n_tokens > 0)
                         ? (size_t)coherence[idx].n_tokens : 1;
        size_t n_bytes = n_tokens * 4;

        wubu_absorb_record_t *rec = &planner->records[planner->n_records];
        strncpy(rec->path, path, sizeof(rec->path) - 1);
        rec->path[sizeof(rec->path) - 1] = '\0';
        rec->density = density;
        rec->coherence = score;
        rec->n_tokens = n_tokens;
        rec->absorbed = 0;

        if (density >= planner->cfg.absorb_threshold &&
            n_absorbed < planner->cfg.absorb_batch &&
            weight_used + n_bytes <= planner->cfg.weight_budget) {
            rec->absorbed = 1;
            n_absorbed++;
            weight_used += n_bytes;
            planner->absorbed_bytes += n_bytes;
            if (planner->model)
                wubu_scalable_mark_hot(planner->model, path);
        } else if (density >= planner->cfg.keep_threshold) {
            planner->kv_bytes += n_bytes;
        } else {
            planner->pruned_bytes += n_bytes;
        }

        planner->n_records++;
    }

    /* Step 5: feed utilization to shrink operator */
    if (planner->shrink && n_files > 0) {
        float *util = (float *)malloc((size_t)n_files * sizeof(float));
        if (util) {
            for (int i = 0; i < n_files; i++)
                util[i] = coherence[i].attention_mass;
            wubu_kv_shrink_feed(planner->shrink, paths, util, n_files);
            free(util);
        }
    }

    if (out_absorbed) *out_absorbed = planner->records;
    if (out_n_absorbed) *out_n_absorbed = planner->n_records;

    return n_absorbed;
}

void wubu_density_planner_stats(wubu_density_planner_t *planner,
                                 size_t *out_absorbed_bytes,
                                 size_t *out_kv_bytes,
                                 size_t *out_pruned_bytes) {
    if (!planner) {
        if (out_absorbed_bytes) *out_absorbed_bytes = 0;
        if (out_kv_bytes) *out_kv_bytes = 0;
        if (out_pruned_bytes) *out_pruned_bytes = 0;
        return;
    }
    if (out_absorbed_bytes) *out_absorbed_bytes = planner->absorbed_bytes;
    if (out_kv_bytes) *out_kv_bytes = planner->kv_bytes;
    if (out_pruned_bytes) *out_pruned_bytes = planner->pruned_bytes;
}
