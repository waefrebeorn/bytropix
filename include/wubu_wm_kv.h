/*
 * wubu_wm_kv.h -- Bounded working-memory KV (O05) + online roofline (N02) +
 * per-layer compute floor (N08). Opaque structs where stateful.
 */
#ifndef WUBU_WM_KV_H
#define WUBU_WM_KV_H

#include <stddef.h>

/* ---- bounded working-memory KV ring (O05) ---- */
typedef struct wubu_wm_kv wubu_wm_kv_t;
wubu_wm_kv_t *wubu_wm_kv_create(int cap);
/* Push token; if over capacity evicts oldest; returns evicted token id or -1. */
int wubu_wm_kv_push(wubu_wm_kv_t *w, int token_id);
int wubu_wm_kv_count(const wubu_wm_kv_t *w);
int wubu_wm_kv_cap(const wubu_wm_kv_t *w);
void wubu_wm_kv_destroy(wubu_wm_kv_t *w);

/* ---- online roofline sampler (N02) ---- */
typedef struct wubu_roofline wubu_roofline_t;
wubu_roofline_t *wubu_roofline_create(double init_beta, double alpha);
void wubu_roofline_observe(wubu_roofline_t *r, double bytes, double secs);
double wubu_roofline_beta(const wubu_roofline_t *r);
void wubu_roofline_destroy(wubu_roofline_t *r);

/* ---- per-layer compute budget floor (N08), [min_floor,1] ---- */
float wubu_layer_floor(int layer, int L, float min_floor);

#endif /* WUBU_WM_KV_H */
