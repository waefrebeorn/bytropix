/*
 * wubu_cla.c — Cross-Layer Attention KV sharing (Round-3 #223/#224/#227).
 * C11, self-contained. CLA reduces KV cache by sharing K/V tensors across
 * groups of layers (factor k): only the group head computes K/V projections;
 * other layers in the group reuse. Attention-type-matched (sliding shares
 * sliding, global shares global). Memory model: ~1/k reduction + GQA.
 */
#include "wubu_cla.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

struct wubu_cla {
    int n_layers;
    int share_k;        /* sharing factor */
    int *kv_owner;      /* per layer: which layer supplies its KV (-1 = self) */
};

/* Plan: assign each layer a KV owner. CLA shares KV across groups of `share_k`
 * *consecutive same-type* layers (attention-type-matched, back-loaded): inside
 * each contiguous run of identical attention type, only the first layer of every
 * `share_k` block computes K/V; the rest reuse. This is the correct Gemma-4 /
 * CLA behavior (a sliding layer must not share a global layer's KV). */
wubu_cla_t *wubu_cla_plan(int n_layers, int share_k, const int *type) {
    if (n_layers <= 0 || share_k <= 0) return NULL;
    wubu_cla_t *c = (wubu_cla_t *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->n_layers = n_layers; c->share_k = share_k;
    c->kv_owner = (int *)malloc(sizeof(int) * n_layers);
    if (!c->kv_owner) { free(c); return NULL; }
    int run_start = 0;
    for (int i = 0; i <= n_layers; i++) {
        /* segment boundary at type change or end */
        if (i == n_layers || (i > 0 && type[i] != type[i-1])) {
            /* assign owners within [run_start, i) */
            for (int j = run_start; j < i; j++) {
                int blk = (j - run_start) / share_k;
                int head = run_start + blk * share_k;
                c->kv_owner[j] = head;   /* shares from block head (same type) */
            }
            run_start = i;
        }
    }
    return c;
}

void wubu_cla_free(wubu_cla_t *c) {
    if (!c) return;
    free(c->kv_owner); free(c);
}

int wubu_cla_kv_owner(const wubu_cla_t *c, int layer) {
    if (!c || layer < 0 || layer >= c->n_layers) return -1;
    return c->kv_owner[layer];
}

/* Fraction of layers that compute their own KV (rest share). */
double wubu_cla_unique_kv_frac(const wubu_cla_t *c) {
    if (!c) return 0;
    int uniq = 0;
    for (int i = 0; i < c->n_layers; i++) if (c->kv_owner[i] == i) uniq++;
    return (double)uniq / c->n_layers;
}

/* Approx KV cache bytes saved vs full-KV, given per-layer KV bytes `kv_bytes`
 * (already GQA-compressed). Returns multiplier reduction, e.g. 0.5 = half. */
double wubu_cla_kv_reduction(const wubu_cla_t *c, double kv_bytes) {
    (void)kv_bytes;
    double frac = wubu_cla_unique_kv_frac(c);
    return 1.0 - frac;   /* fraction of layers NOT storing own KV */
}
