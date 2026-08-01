/*
 * wubu_kv_compress.c -- Attention-score-driven KV compression (L07 SnapKV /
 * L09 CIA). Self-contained C11.
 *
 * Convergence (SnapKV 2404.10519 + CIA + H2O 7-hop): at long context most KV
 * slots carry little attention; retaining the *attention-mass-weighted* subset
 * preserves quality while shrinking the cache. SnapKV clusters the keys and
 * keeps the most-attended cluster; CIA compresses by per-head attention score.
 * Both reduce to: given per-slot cumulative attention scores and a target keep
 * fraction, return the set of slots to retain. This module implements that
 * selector + a cluster variant, pure and testable. The caller applies the
 * remap (like wubu_stream_kv) to actually drop slots.
 *
 * Triple-DA: n<=0 / keep_frac out of [0,1] / null handled; deterministic.
 */
#include "wubu_kv_compress.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Keep the top `keep_frac` of slots by cumulative attention score. Writes the
 * retained slot indices (ascending by score rank) into out_ids (caller-sized
 * >= k). Returns number retained. keep_frac<=0 -> 0; >=1 -> all n. */
int wubu_kv_keep_top_score(const float *scores, int n, float keep_frac,
                           int *out_ids) {
    if (!scores || n <= 0 || !out_ids) return 0;
    if (keep_frac >= 1.0f) { for (int i = 0; i < n; i++) out_ids[i] = i; return n; }
    if (keep_frac <= 0.0f) return 0;
    int k = (int)(keep_frac * n + 1e-6f);
    if (k > n) k = n;
    if (k <= 0) return 0;

    /* selection sort of the top-k scores (n bounded) */
    uint8_t taken[WUBU_KV_COMPRESS_MAX];
    if (n > WUBU_KV_COMPRESS_MAX) n = WUBU_KV_COMPRESS_MAX;
    memset(taken, 0, (size_t)n);
    int retained = 0;
    for (int c = 0; c < k; c++) {
        int best = -1; float best_s = -1e30f;
        for (int i = 0; i < n; i++) {
            if (taken[i]) continue;
            if (scores[i] > best_s) { best_s = scores[i]; best = i; }
        }
        if (best < 0) break;
        taken[best] = 1;
        out_ids[retained++] = best;
    }
    return retained;
}

/* SnapKV-style cluster keep: partition slots into `nclusters` contiguous
 * clusters, compute each cluster's mean attention, keep the top `keep_clusters`
 * clusters (by mean score), return all slots within them. Writes retained slot
 * indices into out_ids (caller-sized >= n). Returns number retained. */
int wubu_kv_keep_clusters(const float *scores, int n, int nclusters,
                         int keep_clusters, int *out_ids) {
    if (!scores || n <= 0 || !out_ids || nclusters <= 0) return 0;
    if (nclusters > n) nclusters = n;
    if (keep_clusters <= 0) return 0;
    if (keep_clusters > nclusters) keep_clusters = nclusters;

    /* per-cluster mean score */
    float cmean[WUBU_KV_COMPRESS_MAX];
    int csz = (n + nclusters - 1) / nclusters;
    for (int c = 0; c < nclusters; c++) {
        int s = c * csz, e = s + csz; if (e > n) e = n;
        float sum = 0.0f; int cnt = 0;
        for (int i = s; i < e; i++) { sum += scores[i]; cnt++; }
        cmean[c] = cnt ? sum / cnt : 0.0f;
    }
    /* select top keep_clusters by mean */
    uint8_t ctaken[WUBU_KV_COMPRESS_MAX];
    memset(ctaken, 0, (size_t)nclusters);
    for (int kc = 0; kc < keep_clusters; kc++) {
        int best = -1; float best_m = -1e30f;
        for (int c = 0; c < nclusters; c++) {
            if (ctaken[c]) continue;
            if (cmean[c] > best_m) { best_m = cmean[c]; best = c; }
        }
        if (best < 0) break;
        ctaken[best] = 1;
    }
    /* emit all slots in kept clusters */
    int retained = 0;
    for (int c = 0; c < nclusters; c++) {
        if (!ctaken[c]) continue;
        int s = c * csz, e = s + csz; if (e > n) e = n;
        for (int i = s; i < e; i++) out_ids[retained++] = i;
    }
    return retained;
}

/* L08 PyramidKV: pyramidal accumulation. Earlier layers keep *more* KV (they
 * are pooled/global); deeper layers keep less (pyramid narrows). Given per-layer
 * keep_frac and the layer depth (0=shallow), returns the adjusted keep_frac so
 * shallow layers retain a larger fraction. depth_frac in [0,1] (0=shallow). */
float wubu_pyramid_keep(float base_keep, float depth_frac, float pyramid) {
    if (base_keep < 0.0f) base_keep = 0.0f;
    if (base_keep > 1.0f) base_keep = 1.0f;
    if (depth_frac < 0.0f) depth_frac = 0.0f;
    if (depth_frac > 1.0f) depth_frac = 1.0f;
    if (pyramid <= 0.0f) pyramid = 1.0f;
    /* shallow (depth_frac=0) -> base_keep * pyramid; deep -> base_keep / pyramid */
    float f = base_keep * (pyramid * (1.0f - depth_frac) + (1.0f / pyramid) * depth_frac);
    if (f < 0.0f) f = 0.0f;
    if (f > 1.0f) f = 1.0f;
    return f;
}
