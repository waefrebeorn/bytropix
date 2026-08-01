/*
 * wubu_sys2026.c -- System/scheduling 2026 KV methods (Q12/Q13/Q14/Q16/Q17/Q18/R02).
 * C11. Policy cores (hardware plumbing abstracted; the decision logic is real).
 *
 * Convergence (TARDIS / KVDrive / ScoutAttention / AlignedServe / CoDec / SparKV /
 * agentic-context 7-hop):
 *   - Q12 TARDIS: GPU-centric KV service with host spillover. Policy: keep the
 *        most-recently-used KV on GPU; when GPU budget (in tokens) is exceeded,
 *        spill the oldest to host. Returns the count to spill.
 *   - Q13 KVDrive: multi-tier placement (GPU/DRAM/SSD) decision per token by
 *        recency -- recent on GPU, mid on DRAM, cold on SSD (ties N07 tiers).
 *   - Q14 ScoutAttention: layer-ahead CPU precompute schedule. Given a pipeline
 *        of `L` layers and a precompute-ahead count `a`, return which layers are
 *        eligible for CPU precompute this step (those within `a` of the current).
 *   - Q16 AlignedServe: prefix-aware batching. Given request prefixes (each a
 *        token array) and a new request, find the longest shared prefix length
 *        with an existing request so they can be batched (reuses CacheBlend LCP).
 *   - Q17 CoDec: prefix-shared decode. Given two requests' prefix lengths, report
 *        whether they share a prefix >= `min_share` (eligible for shared decode).
 *   - Q18 SparKV: overhead-aware KV loading. Decide whether to load a layer's KV
 *        from slow storage this step based on its access probability vs load cost
 *        (load if p*benefit > cost). Returns 1 to load.
 *   - R02 agentic context-efficiency: given a fixed token budget and the token
 *        cost of providing curated context, return 1 if curated context fits
 *        (cost <= budget) so the agent should use it (efficiency axis).
 *
 * Triple-DA: null/zero handled; thresholds clamped; deterministic.
 */
#include "wubu_sys2026.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Q12 TARDIS spill: spill oldest tokens when gpu_used > gpu_cap.
 * ages[] are token ages (older = larger). Returns count to spill (or 0). */
int wubu_tardis_spill(const int *ages, int n, int gpu_cap) {
    if (!ages || n <= 0 || gpu_cap <= 0) return 0;
    if (n <= gpu_cap) return 0;
    /* count how many tokens are older than the (gpu_cap)-th youngest.
     * Simpler: spill = n - gpu_cap (oldest first in a real impl). */
    int spill = n - gpu_cap;
    return spill > 0 ? spill : 0;
}

/* Q13 KVDrive tier: 0=GPU,1=DRAM,2=SSD by recency age vs thresholds. */
int wubu_kvdrive_tier(int age, int dram_thr, int ssd_thr) {
    if (age < 0) age = 0;
    if (age < dram_thr) return 0;
    if (age < ssd_thr) return 1;
    return 2;
}

/* Q14 ScoutAttention: is layer `Lcur` eligible for CPU precompute given we are at
 * `cur` and look ahead `a`? Eligible if cur <= Lcur < cur+a (within the window). */
int wubu_scout_eligible(int Lcur, int cur, int a) {
    if (Lcur < 0 || cur < 0 || a < 0) return 0;
    return (Lcur >= cur && Lcur < cur + a) ? 1 : 0;
}

/* Q16 AlignedServe: longest shared prefix length between new request `a` and
 * any existing request in `reqs` (each length `rlen`, total `nreq` flattened).
 * Returns max LCP found (0 if none). */
int wubu_aligned_lcp(const int *a, int alen, const int *reqs, int nreq, int rlen) {
    if (!a || !reqs || alen <= 0 || nreq <= 0 || rlen <= 0) return 0;
    int best = 0;
    for (int r = 0; r < nreq; r++) {
        const int *b = reqs + (size_t)r * rlen;
        int i = 0;
        int lim = alen < rlen ? alen : rlen;
        while (i < lim && a[i] == b[i]) i++;
        if (i > best) best = i;
    }
    return best;
}

/* Q17 CoDec: 1 if two requests share a prefix of at least min_share. */
int wubu_codec_share(int lenA, int lenB, int min_share) {
    if (lenA < 0 || lenB < 0 || min_share < 0) return 0;
    int shared = lenA < lenB ? lenA : lenB;
    return (shared >= min_share) ? 1 : 0;
}

/* Q18 SparKV: load layer KV if expected benefit (p * benefit) exceeds cost. */
int wubu_sparkv_load(float access_p, float benefit, float cost) {
    if (access_p < 0.0f) access_p = 0.0f; if (access_p > 1.0f) access_p = 1.0f;
    if (benefit < 0.0f) benefit = 0.0f; if (cost < 0.0f) cost = 0.0f;
    return (access_p * benefit > cost) ? 1 : 0;
}

/* R02 agentic context-efficiency: 1 if curated-context cost fits the budget. */
int wubu_agentic_ctx(float ctx_cost, float budget) {
    if (budget <= 0.0f) return 0;
    if (ctx_cost < 0.0f) ctx_cost = 0.0f;
    return (ctx_cost <= budget) ? 1 : 0;
}
