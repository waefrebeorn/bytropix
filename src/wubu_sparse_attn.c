/*
 * wubu_sparse_attn.c -- Block-sparse attention pattern generators
 * (L11 NSA / L12 MoBA). Self-contained C11.
 *
 * Convergence (NSA 2502.11089 + MoBA 2402.13169 7-hop): full attention is
 * O(n^2) and KV-bandwidth-bound at long context; block-sparse attention keeps
 * only a fixed number of blocks per query (NSA: compressed + window + top-k
 * blocks selected by a lightweight gate; MoBA: KV partitioned into segments,
 * each query attends the top-k nearest segments). Both reduce to: given a
 * score per (query-block, kv-block) pair, select the top-k blocks per query and
 * emit a boolean keep-mask. This module generates that mask; the caller applies
 * it during attention (skipping unselected blocks => sub-linear KV traffic).
 *
 * Triple-DA: nblk<=0 / k<=0 / null handled; mask buffer sized nblk*nblk;
 * deterministic (ties broken by lower index).
 */
#include "wubu_sparse_attn.h"
#include <stdlib.h>
#include <string.h>

/* Emit a keep-mask for block-sparse attention. scores[qi*nblk + ki] is the
 * gate/importance of kv-block ki for query-block qi. For each qi, keep the top
 * `k` kv-blocks (k<=nblk). mask is nblk*nblk bytes (row-major, 1=keep). Returns
 * 0 on success, -1 on bad input. */
int wubu_block_sparse_mask(const float *scores, int nblk, int k,
                           uint8_t *mask) {
    if (!scores || !mask || nblk <= 0 || k <= 0) return -1;
    if (k > nblk) k = nblk;
    memset(mask, 0, (size_t)nblk * nblk);
    uint8_t row_used[512];
    int lim = nblk < 512 ? nblk : 512;
    for (int qi = 0; qi < nblk; qi++) {
        memset(row_used, 0, (size_t)lim);
        for (int c = 0; c < k; c++) {
            int best = -1; float best_s = -1e30f;
            for (int ki = 0; ki < nblk; ki++) {
                if (row_used[ki]) continue;
                float s = scores[qi * nblk + ki];
                if (s > best_s) { best_s = s; best = ki; }
            }
            if (best < 0) break;
            row_used[best] = 1;
            mask[qi * nblk + best] = 1;
        }
    }
    return 0;
}

/* MoBA-style segment top-k (L12): partition KV into `nseg` equal segments;
 * scores[qi*nseg + si] is query-block qi's affinity to segment si. Keep top-k
 * segments per query. Writes segment-keep flags[qi*nseg + si] (1=keep). Returns
 * 0 on success, -1 on bad input. */
int wubu_moba_topk(const float *scores, int nq, int nseg, int k,
                   uint8_t *flags) {
    if (!scores || !flags || nq <= 0 || nseg <= 0 || k <= 0) return -1;
    if (k > nseg) k = nseg;
    memset(flags, 0, (size_t)nq * nseg);
    uint8_t used[512];
    int lim = nseg < 512 ? nseg : 512;
    for (int qi = 0; qi < nq; qi++) {
        memset(used, 0, (size_t)lim);
        for (int c = 0; c < k; c++) {
            int best = -1; float best_s = -1e30f;
            for (int si = 0; si < nseg; si++) {
                if (used[si]) continue;
                float s = scores[qi * nseg + si];
                if (s > best_s) { best_s = s; best = si; }
            }
            if (best < 0) break;
            used[best] = 1;
            flags[qi * nseg + best] = 1;
        }
    }
    return 0;
}
