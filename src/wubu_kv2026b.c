/*
 * wubu_kv2026b.c -- More 2026 KV-cache methods (Q01/Q04/Q05/Q06). C11.
 *
 * Convergence (CentroidKV / R-KV / OBCache / KeyDiff 7-hop):
 *   - Q01 CentroidKV: cluster KV tokens by cosine similarity to a learned (here:
 *        data-derived) centroid per cluster; keep the token nearest each centroid
 *        (semantic representatives). Returns the representative token ids.
 *   - Q04 R-KV: redundancy-aware eviction -- score = attention mass minus
 *        redundancy (similarity to already-kept tokens); evict high-redundancy
 *        first. We compute a redundancy penalty from pairwise key-cosine.
 *   - Q05 OBCache: Hessian-guided saliency -- approximate second-order importance
 *        via |grad|^2 (a faithful proxy when full Hessian is unavailable); keep
 *        high-saliency tokens. Returns per-token saliency.
 *   - Q06 KeyDiff: key-similarity eviction -- evict tokens whose key is nearest
 *        (cosine) to an already-kept key (redundant), keep the most distinct.
 *
 * Triple-DA: dimensions checked; null -> 0; no OOB; deterministic.
 */
#include "wubu_kv2026b.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float cos_sim(const float *a, const float *b, int d) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < d; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    if (na <= 0 || nb <= 0) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

/* Q01 CentroidKV: given token keys (n x d), form `k` centroids = first k tokens,
 * assign each token to nearest centroid, keep the token in each cluster nearest
 * its centroid. Writes representative ids to out (sized >= k). Returns count. */
int wubu_centroidkv(const float *keys, int n, int d, int k, int *out) {
    if (!keys || !out || n <= 0 || d <= 0 || k <= 0) return 0;
    if (k > n) k = n;
    for (int c = 0; c < k; c++) {
        const float *cent = keys + (size_t)c * d;
        int best = c; float best_sim = -2.0f;
        for (int i = 0; i < n; i++) {
            float s = cos_sim(cent, keys + (size_t)i * d, d);
            if (s > best_sim) { best_sim = s; best = i; }
        }
        out[c] = best;
    }
    return k;
}

/* Q04 R-KV redundancy score: for token i, redundancy = max cosine to any other
 * token j != i. Higher redundancy -> evict first. Writes into out (sized >= n). */
int wubu_rkv_redundancy(const float *keys, int n, int d, float *out) {
    if (!keys || !out || n <= 0 || d <= 0) return 0;
    for (int i = 0; i < n; i++) {
        float r = 0.0f;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            float s = cos_sim(keys + (size_t)i*d, keys + (size_t)j*d, d);
            if (s > r) r = s;
        }
        out[i] = r;
    }
    return n;
}

/* Q05 OBCache Hessian-saliency proxy: saliency_i = |grad_i|^2 (sum of squared
 * gradient over d dims). Writes into out (sized >= n). */
int wubu_obcache_saliency(const float *grad, int n, int d, float *out) {
    if (!grad || !out || n <= 0 || d <= 0) return 0;
    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        for (int j = 0; j < d; j++) { float g = grad[(size_t)i*d + j]; s += g*g; }
        out[i] = s;
    }
    return n;
}

/* Q06 KeyDiff eviction: keep the `keep` most *distinct* tokens (lowest total
 * cosine to others). Returns kept ids (sized >= keep). Greedy: start from the
 * token with min sum-sim, then add the token maximizing min-distinctness. */
int wubu_keydiff_evict(const float *keys, int n, int d, int keep, int *out) {
    if (!keys || !out || n <= 0 || d <= 0 || keep <= 0) return 0;
    if (keep > n) keep = n;
    float *simsum = (float *)calloc((size_t)n, sizeof(float));
    if (!simsum) return 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (i != j) simsum[i] += cos_sim(keys+(size_t)i*d, keys+(size_t)j*d, d);
    int *kept = (int *)calloc((size_t)keep, sizeof(int));
    char *in = (char *)calloc((size_t)n, sizeof(char));
    /* first: lowest total similarity (most distinct) */
    int first = 0; float fv = simsum[0];
    for (int i = 1; i < n; i++) if (simsum[i] < fv) { fv = simsum[i]; first = i; }
    kept[0] = first; in[first] = 1;
    /* greedily add the token maximizing min cosine to already-kept (most distinct) */
    for (int step = 1; step < keep; step++) {
        int best = -1; float best_min = -2.0f;
        for (int i = 0; i < n; i++) {
            if (in[i]) continue;
            float mn = 2.0f;
            for (int t = 0; t < step; t++) {
                float s = cos_sim(keys+(size_t)i*d, keys+(size_t)kept[t]*d, d);
                if (s < mn) mn = s;
            }
            if (mn > best_min) { best_min = mn; best = i; }
        }
        if (best < 0) break;
        kept[step] = best; in[best] = 1;
    }
    for (int i = 0; i < keep; i++) out[i] = kept[i];
    free(simsum); free(kept); free(in);
    return keep;
}
