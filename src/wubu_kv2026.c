/*
 * wubu_kv2026.c -- Fresh 2026 KV-cache methods (Q02/Q03/Q07/Q09/Q10). C11.
 *
 * Convergence (2026 KV-cache survey + individual papers 7-hop):
 *   - Q02 ChunkKV: group consecutive KV tokens into semantic chunks, score each
 *        chunk by its mean attention (or importance), evict lowest-scoring chunks
 *        first (chunk-level granularity preserves intra-chunk coherence).
 *   - Q03 KVzip: query-agnostic importance = a token's *reconstruction value*
 *        estimated from its attention variance across heads (high variance ->
 *        high reconstruction value -> keep). Returns per-token keep score.
 *   - Q07 LAVa: layer-wise + head-wise dynamic budget -- each (layer,head) gets a
 *        budgeted keep-count proportional to its attention entropy (sharp heads
 *        keep more). Reuses wubu_layer_kv_budget math.
 *   - Q09 FreeKV: speculative top-k KV retrieval -- given per-block scores, return
 *        the top-k block ids to fetch (ties sparse_attn block selection).
 *   - Q10 TTKV: temporal-tiered placement -- fresh KV in HOT tier, aging KV
 *        demotes to WARM/COLD based on recency (ties N07 tier_advice).
 *
 * Triple-DA: n<=0/k<=0 clamped; null -> 0; no OOB; deterministic.
 */
#include "wubu_kv2026.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Q02 ChunkKV chunk-level eviction: group n tokens into `nchunks` consecutive
 * chunks, score each by mean of `scores`, keep the top `keep` chunks; write the
 * kept token ids into out (caller-sized >= n). Returns kept count. */
int wubu_chunkkv_evict(const float *scores, int n, int nchunks, int keep,
                       int *out) {
    if (!scores || !out || n <= 0 || nchunks <= 0 || keep <= 0) return 0;
    if (nchunks > n) nchunks = n;
    if (keep > nchunks) keep = nchunks;

    /* chunk mean scores */
    float *cscore = (float *)malloc((size_t)nchunks * sizeof(float));
    int *chunk_of = (int *)malloc((size_t)n * sizeof(int));
    if (!cscore || !chunk_of) { free(cscore); free(chunk_of); return 0; }
    int csize = (n + nchunks - 1) / nchunks;
    for (int i = 0; i < n; i++) chunk_of[i] = i / csize;
    for (int c = 0; c < nchunks; c++) cscore[c] = 0.0f;
    int *cnt = (int *)calloc((size_t)nchunks, sizeof(int));
    for (int i = 0; i < n; i++) { cscore[chunk_of[i]] += scores[i]; cnt[chunk_of[i]]++; }
    for (int c = 0; c < nchunks; c++) if (cnt[c] > 0) cscore[c] /= cnt[c];

    /* top-`keep` chunk ids by score (simple selection) */
    int *top = (int *)malloc((size_t)nchunks * sizeof(int));
    for (int i = 0; i < nchunks; i++) top[i] = i;
    /* bubble-ish: keep indices of keep highest cscore */
    for (int i = 0; i < keep; i++) {
        for (int j = i + 1; j < nchunks; j++)
            if (cscore[top[j]] > cscore[top[i]]) { int t = top[i]; top[i] = top[j]; top[j] = t; }
    }
    /* write token ids belonging to kept chunks */
    int m = 0;
    for (int i = 0; i < n; i++)
        for (int k = 0; k < keep; k++)
            if (chunk_of[i] == top[k]) { out[m++] = i; break; }

    free(cscore); free(chunk_of); free(cnt); free(top);
    return m;
}

/* Q03 KVzip query-agnostic importance: reconstruction value = attention variance
 * across `nheads` heads for token i: var = mean(h^2) - mean(h)^2; high -> keep.
 * Writes per-token importance into out (caller-sized >= n). Returns n. */
int wubu_kvzip_importance(const float *attn, int n, int nheads, float *out) {
    if (!attn || !out || n <= 0 || nheads <= 0) return 0;
    for (int i = 0; i < n; i++) {
        float mean = 0.0f, mean2 = 0.0f;
        for (int h = 0; h < nheads; h++) {
            float a = attn[(size_t)i * nheads + h];
            mean += a; mean2 += a * a;
        }
        mean /= nheads; mean2 /= nheads;
        float var = mean2 - mean * mean;
        if (var < 0.0f) var = 0.0f;
        out[i] = var;
    }
    return n;
}

/* Q07 LAVa (layer,head) dynamic budget: keep-count for layer l, head h given
 * their attention entropies e_l, e_h in [0,1] and a global cap `cap`. Sharp
 * (low entropy) -> larger share. Returns keep count in [1, cap]. */
int wubu_lava_budget(float e_layer, float e_head, int cap) {
    if (cap <= 0) return 0;
    if (e_layer < 0.0f) e_layer = 0.0f; if (e_layer > 1.0f) e_layer = 1.0f;
    if (e_head < 0.0f) e_head = 0.0f; if (e_head > 1.0f) e_head = 1.0f;
    /* sharpness = (1 - entropy) per axis; product weights the two. */
    float w = (1.0f - e_layer) * (1.0f - e_head);
    int k = 1 + (int)(w * (cap - 1) + 0.5f);
    if (k < 1) k = 1; if (k > cap) k = cap;
    return k;
}

/* Q09 FreeKV speculative top-k retrieval: return indices of top-k scores. */
int wubu_freekv_topk(const float *scores, int n, int k, int *out) {
    if (!scores || !out || n <= 0 || k <= 0) return 0;
    if (k > n) k = n;
    for (int i = 0; i < k; i++) {
        int best = -1; float bv = -1e30f;
        for (int j = 0; j < n; j++) {
            int used = 0; for (int t = 0; t < i; t++) if (out[t] == j) { used = 1; break; }
            if (!used && scores[j] > bv) { bv = scores[j]; best = j; }
        }
        if (best < 0) break;
        out[i] = best;
    }
    return k;
}

/* Q10 TTKV temporal-tiered placement: tier for a token with age `age` (steps
 * since last use) given warm/cold thresholds. 0=HOT,1=WARM,2=COLD. */
int wubu_ttkv_tier(int age, int warm_thr, int cold_thr) {
    if (age < 0) age = 0;
    if (age < warm_thr) return 0;
    if (age < cold_thr) return 1;
    return 2;
}
