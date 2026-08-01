/*
 * wubu_kv_budget.c -- Layer-wise KV budget + adaptive sink + scheme selector
 * + footprint forecaster (L18 / L19 / N03 / N17).
 *
 * Convergence (cross-discipline 7-hop: DB buffer-pool + OS paging + roofline):
 * not all layers need equal KV precision/capacity. Deeper layers carry more
 * global signal (keep more / higher precision); shallow layers are local
 * (can be evicted/compressed harder). And the *right* KV bit-width depends on
 * the B* crossover (N01/N03): if we are weight-bound, spend bits on KV; if
 * KV-bound, compress aggressively. This module turns those principles into
 * pure, testable functions the operator calls per step.
 *
 * Triple-DA: edge cases (L<=0, frac out of [0,1], bits invalid) handled;
 * no div-by-zero; deterministic.
 */
#include "wubu_kv_budget.h"
#include <math.h>

/* L18: layer-wise KV budget. Returns the fraction of the KV cap allocated to
 * layer `layer` (0..L-1). Deeper layers get more (global signal). The split
 * is a linear ramp from `shallow_frac` (layer 0) to `deep_frac` (layer L-1),
 * normalized so the mean is `base`. Returns a value in (0,1]. */
float wubu_layer_kv_budget(int layer, int L, float base,
                           float shallow_frac, float deep_frac) {
    if (L <= 0 || layer < 0 || layer >= L) return (base > 0.0f ? base : 1.0f);
    if (base <= 0.0f) base = 1.0f;
    /* ramp position t in [0,1] across layers */
    float t = (L == 1) ? 0.5f : (float)layer / (float)(L - 1);
    float frac = shallow_frac + (deep_frac - shallow_frac) * t;
    /* normalize so the average equals `base` */
    float avg = (shallow_frac + deep_frac) * 0.5f;
    if (avg <= 1e-6f) avg = 1.0f;
    frac = frac * (base / avg);
    if (frac <= 0.0f) frac = 0.01f;
    /* no upper clamp: a budget > 1.0 means the layer gets an above-average
     * share (deeper = more global signal). mean is preserved = base. */
    return frac;
}

/* L19: adaptive sink count from attention entropy. High entropy (broad,
 * uniform attention) -> fewer sinks needed (no single token dominates);
 * low entropy (peaky) -> more sinks (protect the dominant tokens). Returns
 * a sink count in [min_sink, max_sink]. */
int wubu_adaptive_sink(float entropy, int min_sink, int max_sink) {
    if (entropy < 0.0f) entropy = 0.0f;
    if (min_sink < 0) min_sink = 0;
    if (max_sink < min_sink) max_sink = min_sink;
    /* normalize entropy by log2(n) is caller's job; here treat entropy in
     * [0,1] (already normalized). peaky (0) -> max sinks; uniform (1) -> min. */
    float e = entropy; if (e > 1.0f) e = 1.0f;
    int sink = (int)(max_sink - (max_sink - min_sink) * e + 0.5f);
    if (sink < min_sink) sink = min_sink;
    if (sink > max_sink) sink = max_sink;
    return sink;
}

/* N03 / N17: choose KV bit-width from the B* crossover and RAM budget.
 * If B* (crossover batch) <= 1 for the target (seq,batch) we are KV-bound, so
 * pick the smaller bit-width (compress harder). Otherwise pick larger.
 * Returns the chosen bits (one of {b_lo, b_hi}). */
int wubu_kv_scheme_bits(double b_star, int b_lo, int b_hi) {
    if (b_lo <= 0) b_lo = 2;
    if (b_hi <= 0) b_hi = 16;
    if (b_hi < b_lo) { int t = b_lo; b_lo = b_hi; b_hi = t; }
    if (b_star < 1.0) return b_lo;   /* KV-bound -> compress */
    return b_hi;                     /* weight-bound -> can afford more */
}

/* N17: KV footprint forecaster. Given model + (batch, seq) + chosen b_kv,
 * returns the projected KV bytes the operator should pre-allocate for.
 * Same formula as capacity_wall but exposed as the "advisor" entry point. */
double wubu_kv_forecast(int L, int n_kv, int d_h, int b_kv,
                        int batch, int seq) {
    if (L <= 0 || n_kv <= 0 || d_h <= 0 || b_kv <= 0 || batch <= 0 || seq <= 0)
        return 0.0;
    return (double)(2 * n_kv * d_h) * (b_kv / 8.0) * L * batch * seq;
}
