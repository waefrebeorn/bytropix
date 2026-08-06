/* wubu_grow_kv.c — the KV-space amoeba grow operator
 *
 * Grows KV blocks in the namespace toward under-coherent files.
 * The coherence diagnosis (passed in by the caller from the post-forward
 * coherence reward) identifies which files the model doesn't understand.
 * The grow operator mounts new KV blocks near those files so the model
 * can attend to expanded context on the next forward.
 *
 * Design: docs/wubu1-hive-mind-plan.md §1 Track B, Phase 5 (AN21).
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_grow_kv.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MAX_UNDERCOHERENT 256

struct wubu_grow_kv {
    wubu_kv_embedding_t *kv;
    wubu_grow_kv_cfg_t    cfg;
    /* ranked list of under-coherent files */
    struct {
        char   path[256];
        float  score;
    } under[MAX_UNDERCOHERENT];
    int n_under;
    /* next grow counter (for unique mount paths) */
    int grow_seq;
    /* total KV blocks allocated */
    uint32_t n_kv_blocks;
};

wubu_grow_kv_cfg_t wubu_grow_kv_default_cfg(void) {
    wubu_grow_kv_cfg_t cfg;
    cfg.coherence_threshold = 0.5;
    cfg.min_score_delta = 0.05;
    cfg.max_kv_blocks = 10000;
    cfg.block_size = 256;
    return cfg;
}

wubu_grow_kv_t *wubu_grow_kv_create(wubu_kv_embedding_t *kv,
                                     const wubu_grow_kv_cfg_t *cfg) {
    if (!kv || !cfg) return NULL;
    wubu_grow_kv_t *g = (wubu_grow_kv_t *)calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->kv = kv;
    g->cfg = *cfg;
    g->n_under = 0;
    g->grow_seq = 0;
    g->n_kv_blocks = 0;
    return g;
}

void wubu_grow_kv_free(wubu_grow_kv_t *g) {
    if (!g) return;
    free(g);
}

/* DIAGNOSE: rank files by coherence (worst first) */
int wubu_grow_kv_diagnose(wubu_grow_kv_t *g,
                           const char **paths, const float *scores,
                           int n_files) {
    if (!g || !paths || !scores || n_files <= 0) return 0;
    g->n_under = 0;
    /* Collect under-coherent files */
    for (int i = 0; i < n_files && g->n_under < MAX_UNDERCOHERENT; i++) {
        if (scores[i] < g->cfg.coherence_threshold) {
            /* Find the KV-relative path (strip /kv/in/ prefix if present) */
            const char *p = paths[i];
            if (strncmp(p, "/kv/in/", 7) == 0) p += 7;
            strncpy(g->under[g->n_under].path, p,
                    sizeof(g->under[g->n_under].path) - 1);
            g->under[g->n_under].path[sizeof(g->under[g->n_under].path) - 1] = '\0';
            g->under[g->n_under].score = scores[i];
            g->n_under++;
        }
    }
    /* Sort by score ascending (worst coherence first) — simple insertion sort */
    for (int i = 1; i < g->n_under; i++) {
        /* Use memcpy since g->under elements are not assignable as a whole
         * to a local of anonymous type. Manual swap via memcpy. */
        char   tmp_path[256];
        float  tmp_score;
        strcpy(tmp_path, g->under[i].path);
        tmp_score = g->under[i].score;
        int j = i - 1;
        while (j >= 0 && g->under[j].score > tmp_score) {
            strcpy(g->under[j + 1].path, g->under[j].path);
            g->under[j + 1].score = g->under[j].score;
            j--;
        }
        strcpy(g->under[j + 1].path, tmp_path);
        g->under[j + 1].score = tmp_score;
    }
    return g->n_under;
}

/* GROW: mount new KV blocks toward the worst-coherent files */
int wubu_grow_kv_grow(wubu_grow_kv_t *g, int max_grow) {
    if (!g || g->n_under == 0 || max_grow <= 0) return 0;
    if (g->n_kv_blocks >= g->cfg.max_kv_blocks) return 0;

    int grown = 0;
    for (int i = 0; i < g->n_under && grown < max_grow; i++) {
        /* Mount a new KV block at /kv/in/<path>/grow<N>
         * The Euclidean attractor pulls it toward the under-coherent
         * file's region — by naming it as a child path, the namespace
         * router (Poincaré distance) places it nearby in the manifold. */
        char grow_path[320];
        snprintf(grow_path, sizeof(grow_path), "/kv/in/%s/grow%d",
                 g->under[i].path, g->grow_seq++);

        /* Allocate 1 block (block_size floats) for this grow */
        uint32_t n_blocks = 1;
        /* Use the embedding's freelist by calling encode_tokens with
         * 1 dummy token (the block is allocated but the content is
         * zero-initialized by the grow operator). */
        uint16_t dummy_token[1] = { 0 };
        int rc = wubu_kv_embedding_encode_tokens(g->kv, grow_path,
                                                  dummy_token, 1);
        if (rc == 0) {
            g->n_kv_blocks += n_blocks;
            grown++;
        }
    }
    return grown;
}

int wubu_grow_kv_shrink(wubu_grow_kv_t *g) {
    if (!g || g->n_kv_blocks == 0) return -1;
    /* Shrink: unmount the most recent grow block.
     * In a full implementation, this would use the BI-score (AN04)
     * to find the lowest-utilization block. For now, LIFO. */
    /* We don't track individual grow paths for unmount — the namespace
     * handles cleanup on free. Mark one block shrunk. */
    g->n_kv_blocks--;
    return 0;
}

int wubu_grow_kv_undercoherent(const wubu_grow_kv_t *g,
                                const char **out_paths, int cap) {
    if (!g || !out_paths || cap <= 0) return 0;
    int n = g->n_under < cap ? g->n_under : cap;
    for (int i = 0; i < n; i++)
        out_paths[i] = g->under[i].path;
    return n;
}
