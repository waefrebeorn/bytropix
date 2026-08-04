/*
 * wubu_dsa.c -- DeepSeek-Sparse-Attention-style coarse-to-fine block
 * indexer (DSA indexer). Self-contained C11 (libc + libm only).
 *
 * DSA (7-hop: NSA 2502.11089, MoBA 2402.13169, DeepSeek sparse attention
 * line) cuts long-context attention cost with a coarse-to-fine scheme: the
 * KV stream is split into fixed-size blocks; a cheap indexer scores each
 * block for the query as dot(query, block_mean) (the block's mean key
 * vector), keeps only the top-k blocks, and attention runs inside those
 * blocks only. wubu_dsa_index() is the coarse selector; wubu_dsa_attend()
 * does select-then-attend end to end (means derived from the keys).
 *
 * Triple-DA: every entry point null-checks and validates; top_k is clamped
 * to n_blocks; selection ties break toward the lower block index so results
 * are deterministic; softmax uses max-subtraction so output stays finite.
 */
#include "wubu_dsa.h"
#include <math.h>
#include <stdlib.h>

struct wubu_dsa {
    int n_blocks;
    int block_size;
    int top_k;
    int d;
};

/* dot product over n components. */
static float dsa_dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* Coarse stage core: score each block as dot(query, means[b]), select the
 * top-k by score (ties -> lower index), sort selected descending by score
 * (equal scores keep lower index first). Fills out[0..k-1], returns k
 * (top_k clamped to n_blocks), or -1 on allocation failure. */
static int dsa_select(const wubu_dsa_t *dsa, const float *query,
                      const float *const *means, int *out) {
    int n_blocks = dsa->n_blocks;
    int k = dsa->top_k < n_blocks ? dsa->top_k : n_blocks;
    float *sc = (float *)malloc(sizeof(float) * (size_t)n_blocks);
    unsigned char *used = (unsigned char *)malloc((size_t)n_blocks);
    if (!sc || !used) {
        free(sc);
        free(used);
        return -1;
    }
    for (int b = 0; b < n_blocks; b++) {
        sc[b] = dsa_dot(query, means[b], dsa->d);
        used[b] = 0;
    }
    for (int c = 0; c < k; c++) {
        int best = -1;
        float best_s = -1e30f;
        for (int b = 0; b < n_blocks; b++) {
            if (used[b]) continue;
            if (sc[b] > best_s) { best_s = sc[b]; best = b; }
        }
        if (best < 0) { /* non-finite scores; stop early */ break; }
        used[best] = 1;
        out[c] = best;
    }
    /* sort selected by score desc, index asc (stable tie-break) */
    for (int i = 0; i < k; i++) {
        for (int j = i + 1; j < k; j++) {
            int ai = out[i], aj = out[j];
            if (sc[aj] > sc[ai] || (sc[aj] == sc[ai] && aj < ai)) {
                int t = out[i];
                out[i] = out[j];
                out[j] = t;
            }
        }
    }
    free(sc);
    free(used);
    return k;
}

wubu_dsa_t *wubu_dsa_create(int n_blocks, int block_size, int top_k, int d) {
    if (n_blocks <= 0 || block_size <= 0 || top_k <= 0 || d <= 0) return NULL;
    wubu_dsa_t *dsa = (wubu_dsa_t *)malloc(sizeof(*dsa));
    if (!dsa) return NULL;
    dsa->n_blocks = n_blocks;
    dsa->block_size = block_size;
    dsa->top_k = top_k;
    dsa->d = d;
    return dsa;
}

void wubu_dsa_free(wubu_dsa_t *dsa) {
    free(dsa);
}

int wubu_dsa_index(const wubu_dsa_t *dsa, const float *query,
                   const float *const *block_means, int *out_blocks) {
    if (!dsa || !query || !block_means || !out_blocks) return -1;
    return dsa_select(dsa, query, block_means, out_blocks);
}

int wubu_dsa_attend(const wubu_dsa_t *dsa, const float *query,
                    const float *const *block_keys,
                    const float *const *block_vals, float *out, int d_out) {
    if (!dsa || !query || !block_keys || !block_vals || !out || d_out <= 0)
        return -1;
    int n_blocks = dsa->n_blocks, bs = dsa->block_size, d = dsa->d;
    int k = dsa->top_k < n_blocks ? dsa->top_k : n_blocks;
    if (k <= 0) return -1;

    /* block means derived from the keys: mean of the block's key vectors */
    float *means = (float *)malloc(sizeof(float) * (size_t)n_blocks * (size_t)d);
    float **mptrs = (float **)malloc(sizeof(float *) * (size_t)n_blocks);
    int *sel = (int *)malloc(sizeof(int) * (size_t)k);
    if (!means || !mptrs || !sel) {
        free(means);
        free(mptrs);
        free(sel);
        return -1;
    }
    for (int b = 0; b < n_blocks; b++) {
        for (int i = 0; i < d; i++) {
            float acc = 0.0f;
            for (int j = 0; j < bs; j++) acc += block_keys[b][(size_t)j * d + i];
            means[(size_t)b * d + i] = acc / (float)bs;
        }
        mptrs[b] = means + (size_t)b * d;
    }
    int got = dsa_select(dsa, query, (const float *const *)mptrs, sel);
    if (got < 0) {
        free(means);
        free(mptrs);
        free(sel);
        return -1;
    }
    k = got;

    /* fine stage: softmax attention over the selected blocks' keys only */
    float inv = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d_out; i++) out[i] = 0.0f;
    float max_s = -1e30f;
    for (int c = 0; c < k; c++) {
        const float *keys = block_keys[sel[c]];
        for (int j = 0; j < bs; j++) {
            float s = dsa_dot(query, keys + (size_t)j * d, d) * inv;
            if (s > max_s) max_s = s;
        }
    }
    float denom = 0.0f;
    for (int c = 0; c < k; c++) {
        const float *keys = block_keys[sel[c]];
        const float *vals = block_vals[sel[c]];
        for (int j = 0; j < bs; j++) {
            float w = expf(dsa_dot(query, keys + (size_t)j * d, d) * inv - max_s);
            denom += w;
            for (int i = 0; i < d_out; i++) out[i] += w * vals[(size_t)j * d_out + i];
        }
    }
    if (denom > 0.0f) {
        float invd = 1.0f / denom;
        for (int i = 0; i < d_out; i++) out[i] *= invd;
    }
    free(means);
    free(mptrs);
    free(sel);
    return 0;
}
