/*
 * wubu_wm_kv.c -- Bounded working-memory KV (O05) + online roofline sampler
 * (N02) + per-layer compute budget floor (N08).
 *
 * Convergence (neuro/Titans 7-hop + I/O survey): the brain keeps a *bounded*
 * working memory (Titans: a fixed-capacity long-term memory module) rather than
 * unbounded context. And the *right* operating point must be *measured* online
 * (N02), not assumed. This module gives the operator three measurable, testable
 * levers:
 *   - wubu_wm_kv:   a fixed-capacity KV ring (the "working memory"); pushes
 *                   beyond capacity evict oldest (bounded => never OOM).
 *   - wubu_roofline_sample: estimate effective bandwidth beta_eff from a
 *                   measured (bytes, seconds) observation; exponentially
 *                   smoothed so the operator tracks drift.
 *   - wubu_layer_floor: minimum compute budget per layer (skip floor) so the
 *                   early-exit router never starves a layer below its floor.
 * Triple-DA: capacity==0 handled, ring wrap correct, no div-by-zero.
 */
#include "wubu_wm_kv.h"
#include <stdlib.h>
#include <string.h>

/* ---- bounded working-memory KV ring (O05) ---- */
struct wubu_wm_kv {
    int cap;        /* max resident slots (bounded) */
    int n;          /* current count */
    int head;       /* next write slot */
    int *slot_id;   /* external token id per slot (or -1) */
};

wubu_wm_kv_t *wubu_wm_kv_create(int cap) {
    if (cap <= 0) return NULL;
    wubu_wm_kv_t *w = (wubu_wm_kv_t *)calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->slot_id = (int *)malloc((size_t)cap * sizeof(int));
    if (!w->slot_id) { free(w); return NULL; }
    for (int i = 0; i < cap; i++) w->slot_id[i] = -1;
    w->cap = cap; w->n = 0; w->head = 0;
    return w;
}

/* Push a token; if over capacity, evict the oldest (bounded memory). Returns
 * the evicted slot's token id, or -1 if none evicted. */
int wubu_wm_kv_push(wubu_wm_kv_t *w, int token_id) {
    if (!w) return -1;
    int evicted = -1;
    if (w->n >= w->cap) {
        /* evict oldest = slot at (head - n + cap) % cap */
        int oldest = (w->head - w->n + w->cap) % w->cap;
        evicted = w->slot_id[oldest];
        w->slot_id[oldest] = token_id;
        w->head = (oldest + 1) % w->cap;
    } else {
        w->slot_id[w->head] = token_id;
        w->head = (w->head + 1) % w->cap;
        w->n++;
    }
    return evicted;
}

int wubu_wm_kv_count(const wubu_wm_kv_t *w) { return w ? w->n : 0; }
int wubu_wm_kv_cap(const wubu_wm_kv_t *w)   { return w ? w->cap : 0; }

void wubu_wm_kv_destroy(wubu_wm_kv_t *w) {
    if (!w) return;
    free(w->slot_id);
    free(w);
}

/* ---- online roofline sampler (N02) ---- */
struct wubu_roofline {
    double beta_eff;   /* smoothed effective bandwidth (bytes/s) */
    double alpha;      /* EMA factor (0..1) */
    int    primed;
};

wubu_roofline_t *wubu_roofline_create(double init_beta, double alpha) {
    wubu_roofline_t *r = (wubu_roofline_t *)calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->beta_eff = init_beta > 0.0 ? init_beta : 1e9;
    r->alpha = (alpha > 0.0 && alpha < 1.0) ? alpha : 0.1;
    r->primed = 0;
    return r;
}

/* Feed one observation (bytes moved, seconds taken). Updates beta_eff. */
void wubu_roofline_observe(wubu_roofline_t *r, double bytes, double secs) {
    if (!r || secs <= 0.0 || bytes <= 0.0) return;
    double beta = bytes / secs;
    if (!r->primed) { r->beta_eff = beta; r->primed = 1; }
    else r->beta_eff = r->alpha * beta + (1.0 - r->alpha) * r->beta_eff;
}

double wubu_roofline_beta(const wubu_roofline_t *r) { return r ? r->beta_eff : 0.0; }

void wubu_roofline_destroy(wubu_roofline_t *r) { free(r); }

/* ---- per-layer compute budget floor (N08) ---- */
/* Minimum compute fraction a layer may receive (skip floor). Deeper layers get
 * a slightly higher floor (they carry global signal). Returns [min_floor,1]. */
float wubu_layer_floor(int layer, int L, float min_floor) {
    if (L <= 0 || layer < 0 || layer >= L) return (min_floor > 0.0f ? min_floor : 0.1f);
    if (min_floor < 0.0f) min_floor = 0.1f;
    if (min_floor > 1.0f) min_floor = 1.0f;
    /* gentle ramp: deeper layers +0.1 of min_floor (capped at 1.0) */
    float f = min_floor * (1.0f + 0.1f * (float)layer / (float)L);
    if (f > 1.0f) f = 1.0f;
    return f;
}
