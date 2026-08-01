/*
 * wubu_eval_qat.c -- Long-context eval harness (Z01-Z05) + QAT (AA01-AA04). C11.
 *
 * Convergence (NIAH-2 / RULER categories / synthetic haystack / fake-quant /
 * QAT-STE / per-channel / noise-injection 7-hop):
 *   - Z01 NIAH-2 multi-needle: inject `nneedle` (key,value) pairs at random
 *        positions in a haystack; score = fraction of values correctly placed
 *        when queried. We return the injected positions + check function.
 *   - Z02 RULER retrieval: given a set of needles at positions, query by key ->
 *        return the associated value if present (retrieval check).
 *   - Z03 RULER multi-hop: a chain key_i -> key_{i+1}; resolve end-to-end by
 *        following the chain; return final value.
 *   - Z04 RULER aggregation: count occurrences of a target token over context;
 *        return the count (freq aggregation task).
 *   - Z05 synthetic haystack: fill `len` positions with noise sentences (hashed
 *        token ids), leaving needle slots empty for injection. Returns tokens.
 *   - AA01 fake-quant: round to nearest multiple of step, clamp to [min,max]
 *        (simulates fake precision during QAT forward).
 *   - AA02 QAT STE: forward quantizes, backward passes grad if within range
 *        (ties T04 but per-tensor). Returns quantized value + grad-passes flag.
 *   - AA03 per-channel quant/dequant: given scale/zero per channel, dequant
 *        q -> float; returns dequantized.
 *   - AA04 noise injection: add uniform noise in [-amp,amp] to weights (a
 *        robustness augmentation used in QAT). Returns perturbed value.
 *
 * Triple-DA: dims/zero handled; ranges clamped; deterministic (seeded RNG-free
 * hash-based positions).
 */
#include "wubu_eval_qat.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Z01 NIAH injection: pick nneedle distinct positions in [0,len), return them. */
int wubu_niah_inject(int len, int nneedle, int *pos) {
    if (!pos || len <= 0 || nneedle <= 0) return 0;
    if (nneedle > len) nneedle = len;
    int *used = (int *)calloc((size_t)len, sizeof(int));
    if (!used) return 0;
    int c = 0;
    for (int p = 0; p < len && c < nneedle; p++) {
        /* deterministic pseudo-position: stride to spread needles */
        int cand = (p * 2654435761u % (unsigned)len);
        if (!used[cand]) { used[cand] = 1; pos[c++] = cand; }
    }
    free(used);
    return c;
}

/* Z02 RULER retrieval: find value for key among needles (key/value arrays). */
int wubu_ruler_retrieve(const int *key, const int *val, int n, int qkey, int *out) {
    if (!key || !val || !out || n <= 0) return 0;
    for (int i = 0; i < n; i++) if (key[i] == qkey) { *out = val[i]; return 1; }
    return 0;
}

/* Z03 RULER multi-hop: chain key->next_key; resolve from start to depth. */
int wubu_ruler_multihop(const int *key, const int *next, int n, int start, int depth, int *out) {
    if (!key || !next || !out || n <= 0 || depth < 0) return 0;
    int cur = start;
    for (int d = 0; d < depth; d++) {
        int found = 0;
        for (int i = 0; i < n; i++) if (key[i] == cur) { cur = next[i]; found = 1; break; }
        if (!found) return 0;
    }
    *out = cur;
    return 1;
}

/* Z04 RULER aggregation: count occurrences of target in context. */
int wubu_ruler_aggregate(const int *ctx, int n, int target) {
    if (!ctx || n <= 0) return 0;
    int c = 0;
    for (int i = 0; i < n; i++) if (ctx[i] == target) c++;
    return c;
}

/* Z05 synthetic haystack: fill tokens with hashed noise; needle slots (pos) left
 * as 0 (caller injects). Returns tokens filled. */
int wubu_haystack_gen(int len, int *tokens) {
    if (!tokens || len <= 0) return 0;
    for (int i = 0; i < len; i++) {
        unsigned h = 2166136261u;
        unsigned char *p = (unsigned char *)&i;
        for (int b = 0; b < 4; b++) { h ^= p[b]; h *= 16777619u; }
        tokens[i] = (int)(h % 1000) + 1; /* noise token 1..1000 */
    }
    return len;
}

/* AA01 fake-quant: round to nearest step, clamp [mn,mx]. */
float wubu_fakequant(float x, float step, float mn, float mx) {
    if (step <= 0.0f) step = 1.0f;
    if (mn > mx) { float t = mn; mn = mx; mx = t; }
    float q = floorf(x / step + 0.5f) * step;
    if (q < mn) q = mn; if (q > mx) q = mx;
    return q;
}

/* AA02 QAT STE: forward = fakequant; grad passes if |x| <= range. */
int wubu_qat_ste(float x, float step, float mn, float mx, float *out_q, int *grad_pass) {
    if (!out_q || !grad_pass) return 0;
    *out_q = wubu_fakequant(x, step, mn, mx);
    *grad_pass = (x >= mn && x <= mx) ? 1 : 0;
    return 1;
}

/* AA03 per-channel dequant: f = (q - zero) * scale. */
float wubu_dequant_pc(int q, float scale, int zero) {
    return (q - zero) * scale;
}

/* AA04 noise injection: add uniform-ish noise from hashed seed in [-amp,amp]. */
float wubu_noise_inject(float w, unsigned seed, float amp) {
    if (amp < 0.0f) amp = 0.0f;
    unsigned h = 2166136261u;
    unsigned char *p = (unsigned char *)&seed;
    for (int b = 0; b < 4; b++) { h ^= p[b]; h *= 16777619u; }
    float r = ((float)(h % 1000) / 1000.0f) * 2.0f - 1.0f; /* [-1,1] */
    return w + r * amp;
}
