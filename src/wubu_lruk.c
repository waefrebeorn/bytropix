/*
 * wubu_lruk.c -- LRU-k KV eviction (O01, cross-discipline DB buffer-pool 7-hop).
 *
 * Convergence (DB buffer-pool eviction 7-hop, via A07b priority eviction): the
 * KV cache is a buffer pool; the right eviction policy is LRU-k (keep the k most
 * recently *used* pages, not just the single last). This is a lightweight LRU-k
 * over KV block ids: touch() records recency; select() returns the k least-
 * recently-used ids to evict. Distinct from wubu_kv_evict (which mixes
 * attention importance) — LRU-k is pure recency, the classic buffer-pool policy.
 *
 * Triple-DA: cap 0 -> NULL; evict count clamped to tracked n; deterministic.
 */
#include "wubu_lruk.h"
#include <stdlib.h>
#include <string.h>

struct wubu_lruk {
    int cap;
    int n;
    int *id;        /* block id per slot */
    uint64_t *ts;   /* last-touch logical clock per slot */
    uint64_t clock;
};

wubu_lruk_t *wubu_lruk_create(int cap) {
    if (cap <= 0) return NULL;
    wubu_lruk_t *e = (wubu_lruk_t *)calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->id = (int *)malloc((size_t)cap * sizeof(int));
    e->ts = (uint64_t *)calloc((size_t)cap, sizeof(uint64_t));
    if (!e->id || !e->ts) { free(e->id); free(e->ts); free(e); return NULL; }
    e->cap = cap; e->n = 0; e->clock = 1;
    return e;
}

void wubu_lruk_touch(wubu_lruk_t *e, int block_id) {
    if (!e) return;
    int i;
    for (i = 0; i < e->n; i++) if (e->id[i] == block_id) break;
    if (i == e->n) {
        if (e->n >= e->cap) return; /* full: ignore new (caller should evict first) */
        e->id[i] = block_id;
        e->n++;
    }
    e->ts[i] = e->clock++;
}

/* Select up to k least-recently-used block ids to evict. */
int wubu_lruk_select(wubu_lruk_t *e, int k, int *out_ids) {
    if (!e || !out_ids || k <= 0 || e->n == 0) return 0;
    if (k > e->n) k = e->n;
    /* selection sort by ts ascending (oldest first) */
    uint8_t taken[1024];
    int lim = e->n < 1024 ? e->n : 1024;
    memset(taken, 0, (size_t)lim);
    int sel = 0;
    for (int c = 0; c < k; c++) {
        int best = -1; uint64_t best_ts = (uint64_t)-1;
        for (int i = 0; i < e->n; i++) {
            if (taken[i]) continue;
            if (e->ts[i] < best_ts) { best_ts = e->ts[i]; best = i; }
        }
        if (best < 0) break;
        taken[best] = 1;
        out_ids[sel++] = e->id[best];
    }
    return sel;
}

int wubu_lruk_count(const wubu_lruk_t *e) { return e ? e->n : 0; }

void wubu_lruk_destroy(wubu_lruk_t *e) {
    if (!e) return;
    free(e->id); free(e->ts); free(e);
}
