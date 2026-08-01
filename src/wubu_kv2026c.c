/*
 * wubu_kv2026c.c -- Remaining 2026 KV methods (Q11/Q19/R04/R05). C11.
 *
 * Convergence (DASH-KV / HeteroCache / reasoning-redundancy / multi-agent 7-hop):
 *   - Q11 DASH-KV: hash-based token-level attention scheduling. We compute a
 *        stable token hash (FNV-1a over the rounded key) and assign each token to
 *        one of `nbuckets` buckets; the scheduler keeps the bucket with the
 *        highest aggregate score (asymmetric hashing skips low-value tokens).
 *   - Q19 HeteroCache: per-head heterogeneous precision choice -- a head with
 *        low attention entropy (sharp) keeps high bits; diffuse heads drop to
 *        fewer bits. Returns bits per head in [bmin,bmax].
 *   - R04 reasoning redundancy profiler: given per-token redundancy (from Q04
 *        R-KV) and a reasoning-step flag, report the mean redundancy of
 *        reasoning tokens (a diagnostic the operator uses to trigger eviction).
 *   - R05 multi-agent KV coherence: average pairwise PolyKV coherence across a
 *        set of agent KV summaries; report mean coherence in [0,1].
 *
 * Triple-DA: null/zero handled; buckets/clamps safe; deterministic hashes.
 */
#include "wubu_kv2026c.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* FNV-1a 32-bit over a float key buffer (bytes). */
static unsigned fnv1a(const float *k, int d) {
    unsigned h = 2166136261u;
    const unsigned char *p = (const unsigned char *)k;
    for (int i = 0; i < d * (int)sizeof(float); i++) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}

/* Q11 DASH-KV: assign each token to a bucket by hash; return the bucket id with
 * the highest sum of `scores` (the one to keep/schedule first). Writes per-token
 * bucket ids to out_bucket (sized >= n). Returns winning bucket id. */
int wubu_dashkv_schedule(const float *keys, int n, int d, const float *scores,
                         int nbuckets, int *out_bucket) {
    if (!keys || !scores || !out_bucket || n <= 0 || d <= 0 || nbuckets <= 0)
        return -1;
    float *bsum = (float *)calloc((size_t)nbuckets, sizeof(float));
    if (!bsum) return -1;
    for (int i = 0; i < n; i++) {
        unsigned h = fnv1a(keys + (size_t)i * d, d);
        int b = (int)(h % (unsigned)nbuckets);
        out_bucket[i] = b;
        bsum[b] += scores[i];
    }
    int win = 0; float wv = bsum[0];
    for (int b = 1; b < nbuckets; b++) if (bsum[b] > wv) { wv = bsum[b]; win = b; }
    free(bsum);
    return win;
}

/* Q19 HeteroCache per-head bits: sharp (low entropy) -> bmax; diffuse -> bmin. */
int wubu_hetero_bits(const float *entropy, int nheads, int bmin, int bmax,
                     int *out) {
    if (!entropy || !out || nheads <= 0 || bmin <= 0) return 0;
    if (bmax < bmin) bmax = bmin;
    for (int h = 0; h < nheads; h++) {
        float e = entropy[h]; if (e < 0.0f) e = 0.0f; if (e > 1.0f) e = 1.0f;
        int bits = bmin + (int)((bmax - bmin) * (1.0f - e) + 0.5f);
        if (bits < bmin) bits = bmin; if (bits > bmax) bits = bmax;
        out[h] = bits;
    }
    return nheads;
}

/* R04 reasoning redundancy profiler: mean of `redundancy[i]` over tokens where
 * is_reasoning[i] != 0. Returns mean in [0,1]; -1 if none. */
float wubu_redundancy_profile(const float *redundancy, const char *is_reasoning,
                              int n) {
    if (!redundancy || !is_reasoning || n <= 0) return -1.0f;
    float sum = 0.0f; int c = 0;
    for (int i = 0; i < n; i++)
        if (is_reasoning[i]) { sum += redundancy[i]; c++; }
    return c ? sum / c : -1.0f;
}

/* R05 multi-agent KV coherence: mean pairwise cosine among agent summaries. */
float wubu_multiagent_coherence(const float *sums, int n_agents, int d) {
    if (!sums || n_agents <= 0 || d <= 0) return 0.0f;
    if (n_agents < 2) return 1.0f;
    double dot = 0.0; int pairs = 0;
    for (int i = 0; i < n_agents; i++)
        for (int j = i + 1; j < n_agents; j++) {
            const float *a = sums + (size_t)i * d, *b = sums + (size_t)j * d;
            double na = 0, nb = 0, d_ = 0;
            for (int k = 0; k < d; k++) { d_ += a[k]*b[k]; na += a[k]*a[k]; nb += b[k]*b[k]; }
            if (na > 0 && nb > 0) { dot += d_ / (sqrt(na)*sqrt(nb)); pairs++; }
        }
    return pairs ? (float)(dot / pairs) : 0.0f;
}
