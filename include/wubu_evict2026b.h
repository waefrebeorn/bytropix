/*
 * wubu_evict2026b.h -- KV eviction frontier, batch 2 (Theme IO). C11.
 * The infrastructure + governance the frontier needs: score
 * normalization, hierarchical tiers, sink reserves, batch grouping,
 * pooling, priority queues, dual-score fusion, decision caching,
 * compaction, per-layer governors, policy selection, telemetry.
 */
#ifndef WUBU_EVICT2026B_H
#define WUBU_EVICT2026B_H

#include <stdint.h>

/* Normalize a raw eviction score across heads (scale-free 0..1). */
float wubu_ev_norm(float raw, float lo, float hi);

/* Sink reserve: the first `sink` positions are never evicted. */
int wubu_ev_sink_reserve(int pos, int sink);

/* Batch grouping: collect indices into one-shot discard sets.
 * Returns the number of batches; each batch's indices are contiguous. */
int wubu_ev_batch_groups(const int *drop, int n, int stride,
                         int *batch_starts, int *batch_counts, int cap);

/* 1D max-pooling over attention (SnapKV-style clustering). */
int wubu_ev_pool(const float *attn, int n, int w, float *out, int out_cap);

/* Retention priority queue: heap of (score, index), O(log n) evict. */
typedef struct {
    float *scores;
    int   *idx;
    int    n, cap;
} wubu_ev_pq_t;
int  wubu_ev_pq_init(wubu_ev_pq_t *q, int cap);
int  wubu_ev_pq_push(wubu_ev_pq_t *q, float score, int idx);
int  wubu_ev_pq_pop_min(wubu_ev_pq_t *q, float *score, int *idx); /* evict */
int  wubu_ev_pq_free(wubu_ev_pq_t *q);

/* Dual score: H2O importance x InfiniPot novelty fusion. */
float wubu_ev_dual(float importance, float novelty, float alpha);

/* Decision cache: reuse an eviction score across decode steps. */
typedef struct { int pos; float score; int valid; } wubu_ev_cache_t;
float wubu_ev_cache_get(wubu_ev_cache_t *c, int pos, float fallback);
void  wubu_ev_cache_put(wubu_ev_cache_t *c, int pos, float score);

/* Hierarchical tiers: hot RAM / warm DRAM / cold NVMe. */
int wubu_ev_tier(float score, float hot_th, float warm_th);

/* Cache compaction: defragment retained pages (move survivors down). */
int wubu_ev_compact(int *retain, int n, int *out, int cap);

/* Policy selector: pick the eviction policy by the block profile. */
int wubu_ev_policy_select(float head_skew, float block_skew);

/* Per-layer budget governor: returns the layer's allowed pages. */
int wubu_ev_layer_budget(int layer, int n_layers, int total_pages);

/* Telemetry: record a drop (token, reason) into the ledger counters. */
typedef struct { long dropped, retained; long sink_kept; } wubu_ev_ledger_t;
void wubu_ev_ledger_record(wubu_ev_ledger_t *l, int dropped, int retained);

#endif
