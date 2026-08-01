/*
 * wubu_pd_serve.c -- Disaggregated prefill/decode serving (AB01-AB06) +
 * dynamic compute / mixture-of-depths (AC01-AC03). C11.
 *
 * Convergence (DistServe / Splitwise / Mooncake / Dynamo / MoD / mixture-of-depths 7-hop):
 *   - AB01 pool split: model has N_prefill + N_decode workers; report that the two
 *        phases scale independently (return 1 if split configured).
 *   - AB02 KV handoff: schedule transfer of a request's KV from prefill to decode
 *        once prefill tokens == prompt_len (ready flag). Returns 1 when ready.
 *   - AB03 pull-based routing: decode pulls a prefill-finished request only if its
 *        own queue is below a high-water mark (drains prefill spikes). Returns 1
 *        if decode can accept.
 *   - AB04 heterogeneous mapping: assign prefill to compute-dense tier, decode to
 *        bandwidth-dense tier. Returns tier ids.
 *   - AB05 KV transfer cost model: transfer_time = kv_bytes / bandwidth; fits TTFT
 *        budget if transfer_time <= ttft_budget. Returns 1 if fits.
 *   - AB06 prefix-aware PD routing: if a new request's prefix hash matches a
 *        cached prefill, reuse it (skip prefill). Returns 1 if cache hit.
 *   - AC01 MoD per-token layer-skip router: gate g in [0,1]; skip layer if g<thr.
 *        Returns 1 if the token should execute this layer.
 *   - AC02 mixture-of-depths capacity: cap active layers per token at `cap`; a
 *        token with `depth` requested layers keeps min(depth,cap). Returns kept.
 *   - AC03 early-exit confidence: exit when confidence >= thr (no more layers).
 *        Returns 1 if exit.
 *
 * Triple-DA: dims/zero handled; thresholds clamped; deterministic.
 */
#include "wubu_pd_serve.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* AB01 pool split configured? */
int wubu_pd_split(int n_prefill, int n_decode) {
    return (n_prefill > 0 && n_decode > 0) ? 1 : 0;
}

/* AB02 KV handoff ready when prefill produced all prompt tokens. */
int wubu_kv_handoff_ready(int prefill_done, int prompt_len) {
    if (prompt_len <= 0) return 0;
    return (prefill_done >= prompt_len) ? 1 : 0;
}

/* AB03 pull-based: decode accepts if its queue < high_water. */
int wubu_pull_route(int decode_qlen, int high_water) {
    if (high_water <= 0) return 0;
    return (decode_qlen < high_water) ? 1 : 0;
}

/* AB04 heterogeneous tier mapping: prefill->compute tier (0), decode->bw tier (1). */
void wubu_hetero_map(int *prefill_tier, int *decode_tier) {
    if (prefill_tier) *prefill_tier = 0;
    if (decode_tier) *decode_tier = 1;
}

/* AB05 KV transfer cost: fits TTFT budget? */
int wubu_kv_xfer_fits(double kv_bytes, double bandwidth, double ttft_budget) {
    if (bandwidth <= 0.0 || ttft_budget < 0.0) return 0;
    double t = kv_bytes / bandwidth;
    return (t <= ttft_budget) ? 1 : 0;
}

/* AB06 prefix-aware cache hit (hash match). */
int wubu_prefix_reuse(unsigned req_hash, unsigned cache_hash) {
    return (req_hash == cache_hash) ? 1 : 0;
}

/* AC01 MoD layer-skip: execute layer if gate >= thr. */
int wubu_mod_execute(float gate, float thr) {
    if (thr < 0.0f) thr = 0.0f; if (thr > 1.0f) thr = 1.0f;
    if (gate < 0.0f) gate = 0.0f; if (gate > 1.0f) gate = 1.0f;
    return (gate >= thr) ? 1 : 0;
}

/* AC02 mixture-of-depths capacity. */
int wubu_mod_capacity(int depth, int cap) {
    if (depth < 0) depth = 0;
    if (cap < 0) cap = 0;
    return (depth < cap) ? depth : cap;
}

/* AC03 early-exit confidence. */
int wubu_early_exit(float conf, float thr) {
    if (thr < 0.0f) thr = 0.0f; if (thr > 1.0f) thr = 1.0f;
    if (conf < 0.0f) conf = 0.0f; if (conf > 1.0f) conf = 1.0f;
    return (conf >= thr) ? 1 : 0;
}
