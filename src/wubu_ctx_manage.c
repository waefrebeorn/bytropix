/*
 * wubu_ctx_manage.c -- Context-window + dispatch auto-managers
 * (L16 elastic context / N07 tiered-cache advisor / N14 MoD router). C11.
 *
 * Convergence (StreamingLLM + MoE/MoD 7-hop): the operating window should
 * *adapt* online (elastic: grow when attention is diffuse, shrink when a few
 * sinks dominate to save KV), the cache should be tiered (hot/warm/cold -> which
 * precision / which tier), and mixture-of-depths routers should be calibrated so
 * the skip rate matches the compute budget. All three are pure policy functions
 * the operator calls per step. Triple-DA: clamps, no div-by-zero, deterministic.
 */
#include "wubu_ctx_manage.h"
#include <stdlib.h>
#include <math.h>

/* L16 Elastic window: given current window W, measured attention-entropy e in
 * [0,1] (low=peaky -> few tokens matter), and bounds [wmin,wmax], return the
 * next window. Peaky attention (e low) -> shrink toward wmin (few sinks needed);
 * diffuse (e high) -> grow toward wmax. Smooths by rate in [0,1]. */
int wubu_elastic_window(int W, float entropy, int wmin, int wmax, float rate) {
    if (wmin < 0) wmin = 0;
    if (wmax < wmin) wmax = wmin;
    if (W < wmin) W = wmin;
    if (W > wmax) W = wmax;
    if (entropy < 0.0f) entropy = 0.0f;
    if (entropy > 1.0f) entropy = 1.0f;
    if (rate < 0.0f) rate = 0.0f;
    if (rate > 1.0f) rate = 1.0f;
    /* target = lerp(wmin, wmax, entropy) */
    float target = (float)wmin + entropy * (float)(wmax - wmin);
    float next = (float)W + rate * (target - (float)W);
    int nw = (int)(next + 0.5f);
    if (nw < wmin) nw = wmin;
    if (nw > wmax) nw = wmax;
    return nw;
}

/* N07 Tiered-cache advisor: given a slot's recency in [0,1] (1=hot) and its
 * cumulative attention mass `attn` in [0,inf), recommend a tier:
 *   0 = HOT  (keep in fast mem, full precision)
 *   1 = WARM (keep, reduced precision)
 *   2 = COLD (offload / compress / evict)
 * Hot+heavy -> HOT; cold or light -> COLDER. */
int wubu_tier_advice(float recency, float attn) {
    if (recency < 0.0f) recency = 0.0f;
    if (recency > 1.0f) recency = 1.0f;
    if (attn < 0.0f) attn = 0.0f;
    float score = recency * 0.6f + (attn > 1.0f ? 1.0f : attn) * 0.4f;
    if (score > 0.66f) return 0;   /* HOT */
    if (score > 0.33f) return 1;   /* WARM */
    return 2;                       /* COLD */
}

/* N14 MoD router calibration: given a target skip-rate `target` in [0,1] and the
 * current measured skip-rate `measured`, return the next gate threshold tau in
 * [0,1] that nudges toward target (higher tau -> more skipping). Clamped. */
float wubu_mod_tau(float tau, float target, float measured, float lr) {
    if (tau < 0.0f) tau = 0.0f;
    if (tau > 1.0f) tau = 1.0f;
    if (target < 0.0f) target = 0.0f;
    if (target > 1.0f) target = 1.0f;
    if (lr < 0.0f) lr = 0.0f;
    if (lr > 1.0f) lr = 1.0f;
    /* if we skip too little (measured < target), raise tau */
    float next = tau + lr * (target - measured);
    if (next < 0.0f) next = 0.0f;
    if (next > 1.0f) next = 1.0f;
    return next;
}
