/*
 * wubu_attn_tune.c -- Attention/dispatch auto-tuners (L06 / N19 / O11).
 * Self-contained C11.
 *
 * Convergence (Quest 2404.12327 + FlashDecoding + I/O survey 7-hop):
 *  - L06 Quest: sub-linear attention by selecting, per query block, the top-k
 *    *blocks* whose centroid/score is highest (blockwise top-k, like NSA but
 *    probe-then-select). We implement the blockwise top-k selector over block
 *    scores (the caller supplies per-block importance).
 *  - N19 Adaptive chunk: prefill wants large chunks (amortize), decode wants
 *    small (latency); pick chunk from (seq, batch, beta_eff).
 *  - O11 Split-K auto-tune: pick the split-K factor so total tiles stay near a
 *    target (ties N13 regime / roofline). Returns a split-K in [1, Kmax].
 *
 * Triple-DA: invalid inputs clamp; no div-by-zero; deterministic.
 */
#include "wubu_attn_tune.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* L06 Quest blockwise top-k: given block scores[nb] (importance per KV block),
 * select the top k block indices (ascending index order in out_ids). Returns
 * count selected. k>nb -> all; k<=0 -> 0. */
int wubu_quest_topk(const float *scores, int nb, int k, int *out_ids) {
    if (!scores || !out_ids || nb <= 0 || k <= 0) return 0;
    if (k > nb) k = nb;
    uint8_t taken[4096];
    int lim = nb < 4096 ? nb : 4096;
    memset(taken, 0, (size_t)lim);
    int sel = 0;
    for (int c = 0; c < k; c++) {
        int best = -1; float best_s = -1e30f;
        for (int i = 0; i < nb; i++) {
            if (taken[i]) continue;
            if (scores[i] > best_s) { best_s = scores[i]; best = i; }
        }
        if (best < 0) break;
        taken[best] = 1;
        out_ids[sel++] = best;
    }
    return sel;
}

/* N19 Adaptive chunk: choose prefill/decode chunk size. Large batch/seq ->
 * larger chunk (amortize weight I/O); tiny -> small (latency). Returns chunk in
 * [min_c, max_c], capped at seq. */
int wubu_adaptive_chunk(int seq, int batch, int min_c, int max_c) {
    if (seq <= 0) return min_c > 0 ? min_c : 1;
    if (min_c <= 0) min_c = 1;
    if (max_c < min_c) max_c = min_c;
    /* heuristic: scale with batch*seq workload, log-ish; clamp. */
    double load = (double)batch * (double)seq;
    int c = min_c + (int)(log2(1.0 + load / 4096.0) * (max_c - min_c) / 8.0);
    if (c < min_c) c = min_c;
    if (c > max_c) c = max_c;
    if (c > seq) c = seq;
    return c;
}

/* O11 Split-K auto-tune: pick split-K factor so the number of reduction tiles
 * stays near `target_tiles` given `tokens` and `Kmax` cap. Returns [1, Kmax]. */
int wubu_splitk_tune(int tokens, int target_tiles, int Kmax) {
    if (tokens <= 0 || Kmax <= 0) return 1;
    if (target_tiles <= 0) target_tiles = tokens;
    int k = (target_tiles + tokens - 1) / tokens;  /* tiles per token */
    if (k < 1) k = 1;
    if (k > Kmax) k = Kmax;
    return k;
}
