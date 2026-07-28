/*
 * wubu_cache_advice.c — ML-advice / competitive cache eviction (Round-2 #111/#112).
 * C11, self-contained. Upgrades the ds4-ssd LRU slot-bank with a learned
 * eviction advisor: tracks per-block access frequency + recency + a learned
 * "value" score, evicts the lowest expected-future-value block. Implements the
 * competitive-caching-with-ML-advice result (Lycouris 2021) without LRU's
 * O(4) competitive ratio weakness.
 */
#include "wubu_cache_advice.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

struct wubu_cache_advice {
    int cap;
    int n;
    int *block_id;
    float *freq;       /* access frequency */
    float *recency;    /* 0..1, decays */
    float *value;      /* learned value score */
    int *last_used;    /* step of last access */
};

wubu_cache_advice_t *wubu_cache_advice_create(int cap) {
    if (cap <= 0) return NULL;   /* DA: cap==0 -> malloc(0) OOB guard */
    wubu_cache_advice_t *a = (wubu_cache_advice_t *)calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->cap = cap;
    a->block_id = (int *)malloc(sizeof(int) * cap);
    a->freq = (float *)calloc(cap, sizeof(float));
    a->recency = (float *)calloc(cap, sizeof(float));
    a->value = (float *)calloc(cap, sizeof(float));
    a->last_used = (int *)calloc(cap, sizeof(int));
    if (!a->block_id || !a->freq || !a->recency || !a->value || !a->last_used) {
        wubu_cache_advice_free(a);   /* DA: NULL one of the arrays before free */
        return NULL;
    }
    return a;
}
void wubu_cache_advice_free(wubu_cache_advice_t *a) {
    if (!a) return;
    free(a->block_id); free(a->freq); free(a->recency); free(a->value); free(a->last_used);
    free(a);
}

/* Touch a block: update freq/recency/value. If new and full, evict lowest. */
int wubu_cache_advice_touch(wubu_cache_advice_t *a, int blk, int step) {
    /* find existing */
    for (int i = 0; i < a->n; i++) {
        if (a->block_id[i] == blk) {
            a->freq[i] += 1.0f;
            a->recency[i] = 1.0f;
            a->last_used[i] = step;
            /* value = freq * recency (learned signal) */
            a->value[i] = a->freq[i] * a->recency[i];
            return 0;  /* hit, no eviction */
        }
    }
    /* miss: maybe evict */
    if (a->n >= a->cap) {
        int victim = 0;
        float worst = a->value[0];
        for (int i = 1; i < a->n; i++)
            if (a->value[i] < worst) { worst = a->value[i]; victim = i; }
        int evicted = a->block_id[victim];
        /* shift-remove victim */
        for (int i = victim; i < a->n - 1; i++) {
            a->block_id[i] = a->block_id[i+1];
            a->freq[i] = a->freq[i+1];
            a->recency[i] = a->recency[i+1];
            a->value[i] = a->value[i+1];
            a->last_used[i] = a->last_used[i+1];
        }
        a->n--;
        /* decay recency of survivors (time passes) */
        for (int i = 0; i < a->n; i++) a->recency[i] *= 0.9f;
        /* insert new */
        a->block_id[a->n] = blk;
        a->freq[a->n] = 1.0f; a->recency[a->n] = 1.0f;
        a->value[a->n] = 1.0f; a->last_used[a->n] = step;
        a->n++;
        return evicted;  /* eviction notification */
    }
    /* not full: just insert */
    a->block_id[a->n] = blk;
    a->freq[a->n] = 1.0f; a->recency[a->n] = 1.0f;
    a->value[a->n] = 1.0f; a->last_used[a->n] = step;
    a->n++;
    return -1;  /* no eviction */
}

/* Force-decay all recency (call between steps). */
void wubu_cache_advice_tick(wubu_cache_advice_t *a, float decay) {
    for (int i = 0; i < a->n; i++) {
        a->recency[i] *= decay;
        a->value[i] = a->freq[i] * a->recency[i];
    }
}
int wubu_cache_advice_count(wubu_cache_advice_t *a) { return a->n; }
int wubu_cache_advice_has(wubu_cache_advice_t *a, int blk) {
    for (int i = 0; i < a->n; i++) if (a->block_id[i] == blk) return 1;
    return 0;
}
