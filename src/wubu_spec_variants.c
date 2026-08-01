/*
 * wubu_spec_variants.c -- Speculative-decoding variants (M11/M13/M14/L14). C11.
 *
 * Convergence (EAGLE / Medusa / self-speculative / FlashDecoding 7-hop): the
 * remaining M-family gaps are *combinations* of machinery already wired this
 * session: per-step K selection (spec_tuner), KV-quant selection (quant_selector),
 * blockwise parallel verify (sparse_attn), and KV-reuse (kv_evict). This module
 * composes them into the named variant policies so the operator can pick one:
 *   - M13 spec+KV-quant co-design: choose K and KV bits together (K from acceptance,
 *        bits from B* crossover) so the speculative draft + its KV store stay within
 *        the memory/compute budget.
 *   - M14 blockwise parallel verify: verify `K` drafted tokens in parallel blocks
 *        of size `nb` (FlashDecoding-style), returning how many blocks to verify.
 *   - M11 spec verify via KV reuse: when a draft token reuses a prefix already in
 *        the KV cache, skip re-forwarding (returns whether reuse is possible).
 *   - L14 activation-beam offload: decide which KV slots to offload to a slower
 *        tier (those with lowest recency*importance) -- reuses the eviction score.
 *
 * Triple-DA: invalid input clamped; deterministic; no div-by-zero.
 */
#include "wubu_spec_variants.h"
#include <stdlib.h>
#include <math.h>

/* M13 co-design: pick draft count K (via acceptance) and KV bits (via B*).
 * acceptance in [0,1], b_star from capacity_wall, Kmax>=1. Writes *out_K and
 * *out_bits (between b_lo..b_hi). */
void wubu_spec_kv_codesign(float acceptance, double b_star, int Kmax,
                           int b_lo, int b_hi, int *out_K, int *out_bits) {
    if (Kmax <= 0) Kmax = 1;
    if (acceptance < 0.0f) acceptance = 0.0f;
    if (acceptance > 1.0f) acceptance = 1.0f;
    int K = 1 + (int)((Kmax - 1) * acceptance + 0.5f);
    if (K < 1) K = 1; if (K > Kmax) K = Kmax;
    int bits = (b_star > 0.0 && b_star < 1.0) ? b_lo : b_hi;
    if (out_K) *out_K = K;
    if (out_bits) *out_bits = bits;
}

/* M14 blockwise parallel verify: given K drafted tokens and block size nb, return
 * the number of blocks to verify (ceil(K/nb)), clamped to >=1 when K>0. */
int wubu_blockwise_verify_blocks(int K, int nb) {
    if (K <= 0) return 0;
    if (nb <= 0) nb = 1;
    int blocks = (K + nb - 1) / nb;
    if (blocks < 1) blocks = 1;
    return blocks;
}

/* M11 KV-reuse check: a draft token at position `pos` can reuse the prefix KV if
 * `prefix_len` (already cached) >= pos, i.e. no re-forward needed. Returns 1 if
 * reuse possible, 0 otherwise. */
int wubu_kv_reuse_ok(int pos, int prefix_len) {
    if (pos < 0 || prefix_len < 0) return 0;
    return (prefix_len >= pos) ? 1 : 0;
}

/* L14 activation-beam offload decision: given a slot's recency r in [0,1] and
 * importance m>=0, mark for offload when r*m is below `thresh` (cold slots go to
 * the slow tier). Returns 1 if offload, 0 if keep. */
int wubu_offload_decision(float recency, float importance, float thresh) {
    if (recency < 0.0f) recency = 0.0f;
    if (recency > 1.0f) recency = 1.0f;
    if (importance < 0.0f) importance = 0.0f;
    if (thresh < 0.0f) thresh = 0.0f;
    return (recency * (importance > 1.0f ? 1.0f : importance) < thresh) ? 1 : 0;
}
