/*
 * wubu_evict2026b.c -- KV eviction frontier, batch 2 (Theme IO). C11.
 */
#include "wubu_evict2026b.h"
#include <stdlib.h>
#include <string.h>

float wubu_ev_norm(float raw, float lo, float hi)
{
    if (hi <= lo) return 0.5f;
    float v = (raw - lo) / (hi - lo);
    return v < 0 ? 0 : (v > 1 ? 1 : v);
}

int wubu_ev_sink_reserve(int pos, int sink)
{
    return pos < sink ? 1 : 0;
}

int wubu_ev_batch_groups(const int *drop, int n, int stride,
                         int *batch_starts, int *batch_counts, int cap)
{
    if (!drop || !batch_starts || !batch_counts || cap <= 0) return -1;
    int nbatch = 0, i = 0;
    while (i < n && nbatch < cap) {
        if (drop[i]) {
            int start = i, count = 0;
            while (i < n && drop[i] && count < stride) { count++; i++; }
            batch_starts[nbatch] = start;
            batch_counts[nbatch] = count;
            nbatch++;
        } else i++;
    }
    return nbatch;
}

int wubu_ev_pool(const float *attn, int n, int w, float *out, int out_cap)
{
    if (!attn || !out || w <= 0) return -1;
    int n_out = (n + w - 1) / w;
    if (n_out > out_cap) n_out = out_cap;
    for (int i = 0; i < n_out; i++) {
        float mx = -1e30f;
        for (int j = 0; j < w; j++) {
            int k = i * w + j;
            if (k < n && attn[k] > mx) mx = attn[k];
        }
        out[i] = mx;
    }
    return n_out;
}

static void pq_sift_down(wubu_ev_pq_t *q, int i)
{
    while (1) {
        int l = 2 * i + 1, r = 2 * i + 2, s = i;
        if (l < q->n && q->scores[l] < q->scores[s]) s = l;
        if (r < q->n && q->scores[r] < q->scores[s]) s = r;
        if (s == i) break;
        float ts = q->scores[i]; q->scores[i] = q->scores[s]; q->scores[s] = ts;
        int ti = q->idx[i]; q->idx[i] = q->idx[s]; q->idx[s] = ti;
        i = s;
    }
}

int wubu_ev_pq_init(wubu_ev_pq_t *q, int cap)
{
    if (!q || cap <= 0) return -1;
    q->scores = (float *)malloc(sizeof(float) * cap);
    q->idx = (int *)malloc(sizeof(int) * cap);
    if (!q->scores || !q->idx) { free(q->scores); free(q->idx); return -1; }
    q->n = 0; q->cap = cap;
    return 0;
}

int wubu_ev_pq_push(wubu_ev_pq_t *q, float score, int idx)
{
    if (!q || q->n >= q->cap) return -1;
    int i = q->n++;
    q->scores[i] = score; q->idx[i] = idx;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (q->scores[p] <= q->scores[i]) break;
        float ts = q->scores[p]; q->scores[p] = q->scores[i]; q->scores[i] = ts;
        int ti = q->idx[p]; q->idx[p] = q->idx[i]; q->idx[i] = ti;
        i = p;
    }
    return 0;
}

int wubu_ev_pq_pop_min(wubu_ev_pq_t *q, float *score, int *idx)
{
    if (!q || q->n == 0) return -1;
    if (score) *score = q->scores[0];
    if (idx) *idx = q->idx[0];
    q->n--;
    if (q->n > 0) {
        q->scores[0] = q->scores[q->n];
        q->idx[0] = q->idx[q->n];
        pq_sift_down(q, 0);
    }
    return 0;
}

int wubu_ev_pq_free(wubu_ev_pq_t *q)
{
    if (!q) return -1;
    free(q->scores); free(q->idx);
    q->scores = NULL; q->idx = NULL; q->n = q->cap = 0;
    return 0;
}

float wubu_ev_dual(float importance, float novelty, float alpha)
{
    return alpha * importance + (1.0f - alpha) * novelty;
}

float wubu_ev_cache_get(wubu_ev_cache_t *c, int pos, float fallback)
{
    if (!c) return fallback;
    if (c->valid && c->pos == pos) return c->score;
    return fallback;
}

void wubu_ev_cache_put(wubu_ev_cache_t *c, int pos, float score)
{
    if (!c) return;
    c->pos = pos; c->score = score; c->valid = 1;
}

int wubu_ev_tier(float score, float hot_th, float warm_th)
{
    if (score >= hot_th) return 0;   /* hot RAM */
    if (score >= warm_th) return 1;  /* warm DRAM */
    return 2;                        /* cold NVMe */
}

int wubu_ev_compact(int *retain, int n, int *out, int cap)
{
    if (!retain || !out || cap <= 0) return -1;
    int k = 0;
    for (int i = 0; i < n && k < cap; i++)
        if (retain[i]) out[k++] = i;
    return k;
}

int wubu_ev_policy_select(float head_skew, float block_skew)
{
    /* head-skewed -> head-adaptive (Ada-KV); block-skewed -> LSH */
    if (head_skew > 0.7f) return 1;
    if (block_skew > 0.7f) return 2;
    return 0;   /* generic importance */
}

int wubu_ev_layer_budget(int layer, int n_layers, int total_pages)
{
    if (n_layers <= 0 || layer < 0 || layer >= n_layers) return 0;
    /* the uniform share; a governor could weight early layers higher */
    int share = total_pages / n_layers;
    return share < 1 ? 1 : share;
}

void wubu_ev_ledger_record(wubu_ev_ledger_t *l, int dropped, int retained)
{
    if (!l) return;
    l->dropped += dropped;
    l->retained += retained;
}
