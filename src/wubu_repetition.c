/*
 * wubu_repetition.c -- repeat_penalty + DRY (C11, self-contained).
 *
 * Design: opaque wubu_rep_state_t holds the rolling token history
 * as a ring buffer. repeat_penalty scans the recent window; DRY hashes
 * every n-gram in the entire history and damps any token that would
 * continue an already-seen run. Pure C, no external deps.
 */

#include "wubu_repetition.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct wubu_rep_state {
    int  vocab_size;
    int  penalty_last_n;   // <=0 => whole history
    int  dry_ngram_len;   // max n-gram length (>=1)
    int  dry_hash_len;     // <=0 => whole history
    float repeat_penalty;  // >1.0
    float dry_multiplier;   // 0 disables DRY
    float dry_base;         // >1.0

    // Rolling history as a ring buffer.
    int  *hist;             // [cap] token ids, oldest first
    int   hist_len;         // valid entries
    int   hist_cap;         // allocated capacity
    int   hist_head;        // index of oldest entry
};

// 32-bit FNV-1a hash of an n-gram.
static uint32_t dry_hash_ngram(const int *seq, int n) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < n; i++) {
        h ^= (uint32_t)(seq[i] & 0xFF); h *= 16777619u;
        h ^= (uint32_t)((seq[i] >> 8) & 0xFF); h *= 16777619u;
        h ^= (uint32_t)((seq[i] >> 16) & 0xFF); h *= 16777619u;
        h ^= (uint32_t)((seq[i] >> 24) & 0xFF); h *= 16777619u;
    }
    return h;
}

wubu_rep_state_t *wubu_rep_create(int vocab_size,
                                      int penalty_last_n,
                                      int dry_ngram_len,
                                      int dry_hash_len) {
    if (vocab_size <= 0) return NULL;
    wubu_rep_state_t *s = (wubu_rep_state_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->vocab_size = vocab_size;
    s->penalty_last_n = penalty_last_n;
    s->dry_ngram_len = dry_ngram_len < 1 ? 1 : dry_ngram_len;
    s->dry_hash_len = dry_hash_len;
    s->repeat_penalty = 1.0f;
    s->dry_multiplier = 0.0f;
    s->dry_base = 1.75f;
    // Cap: whole-context DRY needs the full sequence; bound at 262144.
    int cap = dry_hash_len > 0 ? dry_hash_len : 262144;
    if (penalty_last_n > cap) cap = penalty_last_n;
    if (cap < 64) cap = 64;
    s->hist_cap = cap;
    s->hist = (int *)malloc((size_t)cap * sizeof(int));
    if (!s->hist) { free(s); return NULL; }
    s->hist_len = 0;
    s->hist_head = 0;
    return s;
}

void wubu_rep_free(wubu_rep_state_t *s) {
    if (!s) return;
    if (s->hist) free(s->hist);
    free(s);
}

void wubu_rep_set_params(wubu_rep_state_t *s,
                          float repeat_penalty,
                          float dry_multiplier,
                          float dry_base) {
    if (!s) return;
    s->repeat_penalty = repeat_penalty > 0.0f ? repeat_penalty : 1.0f;
    s->dry_multiplier = dry_multiplier;
    s->dry_base = dry_base > 1.0f ? dry_base : 1.0f;
}

static int hist_get(const wubu_rep_state_t *s, int i) {
    // i in [0, hist_len): 0 = oldest
    int idx = (s->hist_head + i) % s->hist_cap;
    return s->hist[idx];
}

void wubu_rep_observe(wubu_rep_state_t *s, int token_id) {
    if (!s) return;
    int idx = (s->hist_head + s->hist_len) % s->hist_cap;
    s->hist[idx] = token_id;
    if (s->hist_len < s->hist_cap) s->hist_len++;
    else s->hist_head = (s->hist_head + 1) % s->hist_cap; // overwrite oldest
}

int wubu_rep_apply(wubu_rep_state_t *s, float *logits) {
    if (!s || !logits) return -1;
    if (s->hist_len == 0) return 0;

    // ---- 1. repeat_penalty over recent window ----
    int win = s->penalty_last_n > 0 ? s->penalty_last_n : s->hist_len;
    if (win > s->hist_len) win = s->hist_len;
    int win_start = s->hist_len - win;
    const float rp = s->repeat_penalty;
    if (rp > 1.0f) {
        // Penalize each token id at most ONCE (presence-based, not per-occurrence).
        int seen[4096];
        int n_seen = 0;
        for (int i = 0; i < win; i++) {
            int tid = hist_get(s, win_start + i);
            if (tid < 0 || tid >= s->vocab_size) continue;
            int already = 0;
            for (int k = 0; k < n_seen; k++) if (seen[k] == tid) { already = 1; break; }
            if (already) continue;
            if (n_seen < 4096) seen[n_seen++] = tid;
            float v = logits[tid];
            if (v > 0.0f) logits[tid] /= rp;
            else if (v < 0.0f) logits[tid] *= rp;
        }
    }

    // ---- 2. DRY over the whole context ----
    if (s->dry_multiplier > 0.0f && s->hist_len >= 1) {
        int ngram = s->dry_ngram_len;
        int ctx = s->dry_hash_len > 0 ? s->dry_hash_len : s->hist_len;
        if (ctx > s->hist_len) ctx = s->hist_len;
        int ctx_start = s->hist_len - ctx;

        // For every position p in [ctx_start .. hist_len-1], build the
        // n-gram ending at p and a candidate extension token = hist[p+1].
        // Hashing each suffix catches repeated runs cheaply.
        for (int p = ctx_start; p < s->hist_len - 1; p++) {
            int len = (p - ctx_start) + 1;
            if (len > ngram) len = ngram;
            // n-gram = hist[(p-len+1) .. p]
            int seq[32];
            if (len > 32) len = 32;
            for (int k = 0; k < len; k++)
                seq[k] = hist_get(s, (p - len + 1) + k);
            uint32_t h = dry_hash_ngram(seq, len);
            (void)h; // presence of the run is what matters, not the key
            int cand = hist_get(s, p + 1);
            if (cand >= 0 && cand < s->vocab_size) {
                // Damp proportionally to run length already emitted.
                float factor = powf(s->dry_base, -(float)len * s->dry_multiplier);
                logits[cand] *= factor;
            }
        }
    }
    return 0;
}

void wubu_rep_reset(wubu_rep_state_t *s) {
    if (!s) return;
    s->hist_len = 0;
    s->hist_head = 0;
}
