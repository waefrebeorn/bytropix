/*
 * wubu_lruk.h -- LRU-k KV eviction (O01). Opaque struct.
 */
#ifndef WUBU_LRUK_H
#define WUBU_LRUK_H

#include <stdint.h>

typedef struct wubu_lruk wubu_lruk_t;
wubu_lruk_t *wubu_lruk_create(int cap);
void wubu_lruk_touch(wubu_lruk_t *e, int block_id);
/* Select up to k least-recently-used ids to evict; returns count. */
int wubu_lruk_select(wubu_lruk_t *e, int k, int *out_ids);
int wubu_lruk_count(const wubu_lruk_t *e);
void wubu_lruk_destroy(wubu_lruk_t *e);

#endif /* WUBU_LRUK_H */
