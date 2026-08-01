/*
 * wubu_spec_tuner.c -- Speculative decode auto-tuner (N15 / M12 / M15).
 *
 * Convergence (self-speculative + EAGLE 7-hop): the number of draft tokens K
 * should track the *measured* acceptance rate. If acceptance is high, raise K;
 * if low, lower K. And the optimal K can differ per layer (M12). This module
 * maintains a per-layer acceptance estimate and proposes K.
 *
 * Triple-DA: K clamped to [1, Kmax]; acceptance rate in [0,1]; no div-by-zero.
 */
#include "wubu_spec_tuner.h"
#include <stdlib.h>

struct wubu_spec_tuner {
    int L;
    int Kmax;
    float *acc;       /* per-layer smoothed acceptance rate */
    float alpha;
};

wubu_spec_tuner_t *wubu_spec_tuner_create(int L, int Kmax, float alpha) {
    if (L <= 0 || Kmax <= 0) return NULL;
    wubu_spec_tuner_t *t = (wubu_spec_tuner_t *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->acc = (float *)calloc((size_t)L, sizeof(float));
    if (!t->acc) { free(t); return NULL; }
    t->L = L; t->Kmax = Kmax;
    t->alpha = (alpha > 0.0f && alpha < 1.0f) ? alpha : 0.1f;
    for (int i = 0; i < L; i++) t->acc[i] = 0.5f; /* prior: neutral */
    return t;
}

/* Record accepted/total for layer `layer`; updates smoothed acceptance. */
void wubu_spec_tuner_observe(wubu_spec_tuner_t *t, int layer,
                             int accepted, int total) {
    if (!t || layer < 0 || layer >= t->L || total <= 0) return;
    float r = (float)accepted / (float)total;
    if (r < 0.0f) r = 0.0f; if (r > 1.0f) r = 1.0f;
    t->acc[layer] = t->alpha * r + (1.0f - t->alpha) * t->acc[layer];
}

/* Propose draft count K for layer `layer`. Higher acceptance -> larger K,
 * clamped to [1, Kmax]. Target: K ~ 1 + (Kmax-1)*acceptance. */
int wubu_spec_tuner_K(const wubu_spec_tuner_t *t, int layer) {
    if (!t || layer < 0 || layer >= t->L) return 1;
    float a = t->acc[layer];
    if (a < 0.0f) a = 0.0f; if (a > 1.0f) a = 1.0f;
    int K = 1 + (int)((t->Kmax - 1) * a + 0.5f);
    if (K < 1) K = 1;
    if (K > t->Kmax) K = t->Kmax;
    return K;
}

/* N16 cache-hit feedback: track prefix-reuse hit rate to advise whether to
 * keep a prefix cache warm (returns smoothed hit rate in [0,1]). */
struct wubu_cache_fb {
    float hit;     /* smoothed hit rate */
    float alpha;
};

wubu_cache_fb_t *wubu_cache_fb_create(float alpha) {
    wubu_cache_fb_t *c = (wubu_cache_fb_t *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->alpha = (alpha > 0.0f && alpha < 1.0f) ? alpha : 0.1f;
    c->hit = 0.0f;
    return c;
}

void wubu_cache_fb_observe(wubu_cache_fb_t *c, int hits, int queries) {
    if (!c || queries <= 0) return;
    float r = (float)hits / (float)queries;
    if (r < 0.0f) r = 0.0f; if (r > 1.0f) r = 1.0f;
    c->hit = c->alpha * r + (1.0f - c->alpha) * c->hit;
}

float wubu_cache_fb_hitrate(const wubu_cache_fb_t *c) { return c ? c->hit : 0.0f; }

void wubu_spec_tuner_destroy(wubu_spec_tuner_t *t) { if (t) { free(t->acc); free(t); } }
void wubu_cache_fb_destroy(wubu_cache_fb_t *c) { free(c); }
