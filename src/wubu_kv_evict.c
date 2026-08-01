/*
 * wubu_kv_evict.c — Priority-based KV eviction (doc A07b).
 * See header for the policy. Self-contained C11.
 */
#include "wubu_kv_evict.h"
#include <stdlib.h>
#include <string.h>

wubu_kv_evict_t *wubu_kv_evict_create(float decay) {
    wubu_kv_evict_t *e = (wubu_kv_evict_t *)calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->decay = (decay > 0.0f && decay < 1.0f) ? decay : 0.95f;
    return e;
}

void wubu_kv_evict_free(wubu_kv_evict_t *e) {
    free(e);
}

static int find(wubu_kv_evict_t *e, int block_id) {
    for (int i = 0; i < e->n; i++)
        if (e->entries[i].block_id == block_id) return i;
    return -1;
}

void wubu_kv_evict_track(wubu_kv_evict_t *e, int block_id, float importance) {
    if (e->n >= WUBU_KV_EVICT_MAX) return;
    if (find(e, block_id) >= 0) return; /* already tracked */
    wubu_kv_evict_entry_t *en = &e->entries[e->n++];
    en->block_id = block_id;
    en->importance = importance >= 0.0f ? importance : 0.0f;
    en->last_access_ema = 1.0f; /* just created = hot */
    en->present = 1;
}

void wubu_kv_evict_touch(wubu_kv_evict_t *e, int block_id) {
    int i = find(e, block_id);
    if (i >= 0) e->entries[i].last_access_ema = 1.0f;
}

void wubu_kv_evict_tick(wubu_kv_evict_t *e) {
    for (int i = 0; i < e->n; i++) {
        e->entries[i].last_access_ema *= e->decay;
        if (e->entries[i].last_access_ema < 1e-6f) e->entries[i].last_access_ema = 0.0f;
    }
}

void wubu_kv_evict_drop(wubu_kv_evict_t *e, int block_id) {
    int i = find(e, block_id);
    if (i < 0) return;
    e->entries[i] = e->entries[e->n - 1];
    e->n--;
}

float wubu_kv_evict_score(const wubu_kv_evict_entry_t *ent) {
    /* Lower score => evict first. Score combines recency and importance.
     * We use 1/(recency * (1+importance)) so that:
     *   - stale (low recency) -> high score -> evicted first
     *   - important            -> low score  -> retained
     * Add small epsilon to avoid div-by-zero. */
    float denom = (ent->last_access_ema + 1e-6f) * (1.0f + ent->importance);
    return 1.0f / denom;
}

int wubu_kv_evict_select(wubu_kv_evict_t *e, int *out_ids, int out_n) {
    if (!out_ids || out_n <= 0 || e->n == 0) return 0;

    /* Simple selection sort of the lowest-scoring `out_n` entries.
     * n is bounded (<= 4096) and eviction is infrequent, so O(n^2) is fine. */
    int selected = 0;
    uint8_t taken[WUBU_KV_EVICT_MAX];
    memset(taken, 0, (size_t)e->n);

    for (int k = 0; k < out_n && k < e->n; k++) {
        int best = -1;
        float best_score = -1.0f;
        for (int i = 0; i < e->n; i++) {
            if (taken[i]) continue;
            float s = wubu_kv_evict_score(&e->entries[i]);
            if (best < 0 || s > best_score) {
                best = i;
                best_score = s;
            }
        }
        if (best < 0) break;
        taken[best] = 1;
        out_ids[selected++] = e->entries[best].block_id;
    }
    return selected;
}

int wubu_kv_evict_count(const wubu_kv_evict_t *e) {
    return e ? e->n : 0;
}
