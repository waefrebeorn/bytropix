/*
 * wubu_kv_evict.h — Priority-based KV eviction (doc A07b / NVIDIA H2O-style).
 *
 * Evicts the least-important KV blocks first. Importance is a per-block score
 * combining:
 *   - recency  : exponential-decay LRU (EMA of last-access time)
 *   - attention: a per-block importance weight (e.g. accumulated attention
 *                mass, or a fixed per-layer prior). Higher = keep.
 *
 * eviction_score = recency_ema * (1 + importance)
 * Lowest score is evicted first. This is the "importance + LRU" hybrid from
 * doc A07b (NVIDIA priority-based KV eviction). Pure C11, no third-party dep.
 */
#ifndef WUBU_KV_EVICT_H
#define WUBU_KV_EVICT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_KV_EVICT_MAX 4096

typedef struct {
    int    block_id;        /* stable external id for the KV page */
    float  last_access_ema; /* 0..1, decays when not accessed */
    float  importance;      /* attention/importance weight >= 0 */
    uint8_t present;        /* 1 if currently resident */
} wubu_kv_evict_entry_t;

typedef struct {
    wubu_kv_evict_entry_t entries[WUBU_KV_EVICT_MAX];
    int n;                  /* number of tracked entries */
    float decay;            /* EMA decay per step (e.g. 0.95) */
} wubu_kv_evict_t;

/* Create an eviction tracker. decay in (0,1): higher = longer memory. */
wubu_kv_evict_t *wubu_kv_evict_create(float decay);

/* Register/track a block (id must be stable & unique per active page). */
void wubu_kv_evict_track(wubu_kv_evict_t *e, int block_id, float importance);

/* Mark a block accessed this step (bumps recency EMA to 1.0). */
void wubu_kv_evict_touch(wubu_kv_evict_t *e, int block_id);

/* Advance the recency clock by one decode step (decays all EMAs). */
void wubu_kv_evict_tick(wubu_kv_evict_t *e);

/* Drop tracking for a block (e.g. sequence finished). */
void wubu_kv_evict_drop(wubu_kv_evict_t *e, int block_id);

/* Compute eviction priority score for a tracked entry. Higher = evict first
 * when we invert: returns lowest-importance first. We return the raw score;
 * the caller sorts ascending and evicts from the front. */
float wubu_kv_evict_score(const wubu_kv_evict_entry_t *ent);

/* Select up to `out_n` victim block ids to evict (lowest score first).
 * Writes victim ids into out_ids (caller allocates out_n ints).
 * Returns number actually selected (<= out_n). */
int wubu_kv_evict_select(wubu_kv_evict_t *e, int *out_ids, int out_n);

/* Stats: current tracked count. */
int wubu_kv_evict_count(const wubu_kv_evict_t *e);

void wubu_kv_evict_free(wubu_kv_evict_t *e);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_KV_EVICT_H */
