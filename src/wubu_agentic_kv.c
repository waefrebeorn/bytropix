/*
 * wubu_agentic_kv.c -- Hybrid scheduler + multimodal/agentic KV (S06/U01/U02/U03/U04/U05). C11.
 *
 * Convergence (hybrid-scheduler / Gemma-4-shared-KV / DeepSeek-V4-CSA /
 * LMCache-vision-hash / LOOK-M / agentic-compaction 7-hop):
 *   - S06 hybrid layer scheduler: a 3:1 GDN:GA mix. Given layer index L and
 *        period p (e.g. 4), layer L uses recurrent update if (L % p) != 0 else
 *        full attention. Returns 1 for recurrent, 0 for attention.
 *   - U01 shared-KV (Gemma-4): later layers reuse an earlier layer's KV. Given
 *        the sharing offset `off`, layer L's KV is sourced from layer (L-off)
 *        when L >= off. Returns the source layer id (or L if not sharing).
 *   - U02 CSA/HCA compressed entry: fold `group` consecutive tokens into one
 *        compressed KV entry (mean-pool of keys). Writes `n/group` entries.
 *   - U03 LMCache vision-hash: hash a vision token block (FNV) so identical
 *        images dedupe their KV across requests. Returns a 32-bit hash.
 *   - U04 LOOK-M vision prune: keep the top-`keep` vision tokens by attention
 *        importance (score). Returns kept ids.
 *   - U05 agentic memory compaction: given `n` turns each with a saliency, keep
 *        the top-`keep` by saliency, marking the rest compacted (returned as a
 *        bitmask intent: 1 = keep, 0 = compact). Writes into out (sized >= n).
 *
 * Triple-DA: dims/period checked; null handled; deterministic.
 */
#include "wubu_agentic_kv.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* S06 hybrid scheduler: recurrent unless (L % period) == 0 (full attention). */
int wubu_hybrid_is_recurrent(int L, int period) {
    if (L < 0) return 0;
    if (period <= 0) period = 1;
    return ((L % period) == 0) ? 0 : 1;
}

/* U01 shared-KV source layer. */
int wubu_shared_kv_source(int L, int off) {
    if (L < 0 || off <= 0) return L;
    if (L < off) return L;
    return L - off;
}

/* U02 CSA: mean-pool `group` consecutive keys (d-dim) into entries. Returns count. */
int wubu_csa_compress(const float *keys, int n, int d, int group, float *out) {
    if (!keys || !out || n <= 0 || d <= 0 || group <= 0) return 0;
    int cnt = 0;
    for (int i = 0; i < n; i += group) {
        int lim = (i + group < n) ? (i + group) : n;
        for (int j = 0; j < d; j++) {
            float s = 0.0f;
            for (int t = i; t < lim; t++) s += keys[(size_t)t*d + j];
            out[(size_t)cnt*d + j] = s / (lim - i);
        }
        cnt++;
    }
    return cnt;
}

/* U03 vision-block hash (FNV-1a over the token ids). */
unsigned wubu_vision_hash(const int *tok, int n) {
    if (!tok || n <= 0) return 0;
    unsigned h = 2166136261u;
    for (int i = 0; i < n; i++) {
        unsigned v = (unsigned)tok[i];
        unsigned char *p = (unsigned char *)&v;
        for (int b = 0; b < 4; b++) { h ^= p[b]; h *= 16777619u; }
    }
    return h;
}

/* U04 LOOK-M: keep top `keep` vision token ids by descending score. */
int wubu_lookm_keep(const float *score, int n, int keep, int *out) {
    if (!score || !out || n <= 0 || keep <= 0) return 0;
    if (keep > n) keep = n;
    /* simple selection: copy ids, partial-selection sort by score desc */
    int *idx = (int *)malloc((size_t)n * sizeof(int));
    if (!idx) return 0;
    for (int i = 0; i < n; i++) idx[i] = i;
    for (int k = 0; k < keep; k++) {
        int best = k;
        for (int j = k + 1; j < n; j++)
            if (score[idx[j]] > score[idx[best]]) best = j;
        int t = idx[k]; idx[k] = idx[best]; idx[best] = t;
        out[k] = idx[k];
    }
    free(idx);
    return keep;
}

/* U05 agentic compaction: 1=keep (top saliency), 0=compact. Writes mask. */
int wubu_agentic_compact(const float *saliency, int n, int keep, char *out) {
    if (!saliency || !out || n <= 0 || keep <= 0) return 0;
    if (keep > n) keep = n;
    int *idx = (int *)malloc((size_t)n * sizeof(int));
    if (!idx) return 0;
    for (int i = 0; i < n; i++) { idx[i] = i; out[i] = 0; }
    for (int k = 0; k < keep; k++) {
        int best = k;
        for (int j = k + 1; j < n; j++)
            if (saliency[idx[j]] > saliency[idx[best]]) best = j;
        int t = idx[k]; idx[k] = idx[best]; idx[best] = t;
        out[idx[k]] = 1;
    }
    free(idx);
    return keep;
}
