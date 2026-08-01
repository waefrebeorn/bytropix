/*
 * wubu_spec_tuner.h -- Speculative decode auto-tuner (N15/M12/M15) + cache-hit
 * feedback loop (N16). Opaque structs where stateful.
 */
#ifndef WUBU_SPEC_TUNER_H
#define WUBU_SPEC_TUNER_H

/* Per-layer draft-count tuner (N15/M12: pick K from acceptance rate). */
typedef struct wubu_spec_tuner wubu_spec_tuner_t;
wubu_spec_tuner_t *wubu_spec_tuner_create(int L, int Kmax, float alpha);
void wubu_spec_tuner_observe(wubu_spec_tuner_t *t, int layer,
                             int accepted, int total);
int wubu_spec_tuner_K(const wubu_spec_tuner_t *t, int layer);
void wubu_spec_tuner_destroy(wubu_spec_tuner_t *t);

/* Cache-hit feedback loop (N16): smoothed prefix-reuse hit rate. */
typedef struct wubu_cache_fb wubu_cache_fb_t;
wubu_cache_fb_t *wubu_cache_fb_create(float alpha);
void wubu_cache_fb_observe(wubu_cache_fb_t *c, int hits, int queries);
float wubu_cache_fb_hitrate(const wubu_cache_fb_t *c);
void wubu_cache_fb_destroy(wubu_cache_fb_t *c);

#endif /* WUBU_SPEC_TUNER_H */
