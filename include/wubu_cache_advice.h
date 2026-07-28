#ifndef WUBU_CACHE_ADVICE_H
#define WUBU_CACHE_ADVICE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_cache_advice wubu_cache_advice_t;

wubu_cache_advice_t *wubu_cache_advice_create(int cap);
void wubu_cache_advice_free(wubu_cache_advice_t *a);

/* Touch block; returns evicted block id, or -1 if none. */
int wubu_cache_advice_touch(wubu_cache_advice_t *a, int blk, int step);
void wubu_cache_advice_tick(wubu_cache_advice_t *a, float decay);

/* Accessors (struct is opaque). */
int wubu_cache_advice_count(wubu_cache_advice_t *a);
int wubu_cache_advice_has(wubu_cache_advice_t *a, int blk);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_CACHE_ADVICE_H */
