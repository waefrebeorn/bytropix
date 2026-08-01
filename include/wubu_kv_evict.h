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
    int   h2o;              /* 1 = H2O heavy-hitter (attention-mass) mode */
    float attn_sum;         /* running sum of all attention mass fed in */
} wubu_kv_evict_t;

/* Create an eviction tracker. decay in (0,1): higher = longer memory.
 * h2o != 0 enables H2O heavy-hitter mode (L03): eviction is driven by
 * cumulative attention mass (tokens that receive high attention are kept,
 * the rest evicted) instead of recency x importance. */
wubu_kv_evict_t *wubu_kv_evict_create(float decay);

/* Register/track a block (id must be stable & unique per active page). */
void wubu_kv_evict_track(wubu_kv_evict_t *e, int block_id, float importance);

/* H2O (L03): accumulate attention mass for a block. The more attention a
 * token receives, the more of a "heavy hitter" it is and the more it must
 * be retained. attn in [0,1] (e.g. softmax probability it got this step). */
void wubu_kv_evict_track_attn(wubu_kv_evict_t *e, int block_id, float attn);

/* Enable/disable H2O mode. */
void wubu_kv_evict_set_h2o(wubu_kv_evict_t *e, int on);

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

/* H2O (L03): keep the top `keep_frac` (0..1) of blocks by cumulative
 * attention mass; return the victim ids of the rest. Writes up to out_n ids;
 * returns number selected. keep_frac=1.0 => keep all (no eviction). */
int wubu_kv_evict_select_h2o(wubu_kv_evict_t *e, int *out_ids, int out_n,
                             float keep_frac);

/* Stats: current tracked count. */
int wubu_kv_evict_count(const wubu_kv_evict_t *e);

void wubu_kv_evict_free(wubu_kv_evict_t *e);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_KV_EVICT_H */
