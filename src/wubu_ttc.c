/*
 * wubu_ttc.c -- Test-time-compute + multi-agent KV (Q08/Q15/Q20/R01/R03). C11.
 *
 * Convergence (PolyKV / HotPrefix / inference-time-scaling / CATTS 7-hop):
 *   - Q08 PolyKV: a shared, asymmetrically-compressed KV pool across agents.
 *        We model a pool with a coherence check: given two agents' KV summaries
 *        (mean key vector per layer), report whether they share a reusable prefix
 *        (cosine above threshold) -> safe to share (coherence gate).
 *   - Q15 HotPrefix: hotness-aware prefix scheduling -- a prefix with access
 *        count `freq` and recency `age` gets a priority = freq / (1+age*halflife);
 *        higher priority -> schedule/keep first. Reuses LCP (CacheBlend) idea.
 *   - Q20 test-time budget allocator: given a token budget B and per-step cost
 *        estimate, return how many reasoning steps to allow (B / max(1,cost)).
 *   - R01 inference-time scaling controller: pick scaling factor s in [smin,smax]
 *        from a quality-vs-cost tradeoff signal q in [0,1] (higher q -> more
 *        compute). Returns s.
 *   - R03 CATTS contrastive adaptive token scaling: given a draft length and a
 *        contrastive confidence c, shrink the allowed tokens when c is low
 *        (fewer, higher-confidence tokens). Returns allowed tokens.
 *
 * Triple-DA: null/zero handled; thresholds clamped; deterministic.
 */
#include "wubu_ttc.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float coss(const float *a, const float *b, int d) {
    double dot=0,na=0,nb=0;
    for (int i=0;i<d;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
    if (na<=0||nb<=0) return 0.0f;
    return (float)(dot/(sqrt(na)*sqrt(nb)));
}

/* Q08 PolyKV coherence gate: 1 if two agents' KV summaries share a reusable
 * prefix (cosine >= thr). */
int wubu_polykv_coherent(const float *sumA, const float *sumB, int d, float thr) {
    if (!sumA || !sumB || d <= 0) return 0;
    if (thr < 0.0f) thr = 0.0f; if (thr > 1.0f) thr = 1.0f;
    return (coss(sumA, sumB, d) >= thr) ? 1 : 0;
}

/* Q15 HotPrefix priority = freq / (1 + age*halflife). */
float wubu_hotprefix_priority(int freq, int age, float halflife) {
    if (freq < 0) freq = 0; if (age < 0) age = 0;
    if (halflife < 0.0f) halflife = 0.0f;
    return (float)freq / (1.0f + halflife * (float)age);
}

/* Q20 test-time budget allocator: steps = floor(B / max(1,cost)). */
int wubu_ttc_budget_steps(int budget, float cost) {
    if (budget <= 0) return 0;
    if (cost <= 0.0f) cost = 1.0f;
    int s = (int)((float)budget / cost);
    return s < 0 ? 0 : s;
}

/* R01 inference-time scaling controller: s in [smin,smax] from q in [0,1]. */
float wubu_scaling_factor(float q, float smin, float smax) {
    if (q < 0.0f) q = 0.0f; if (q > 1.0f) q = 1.0f;
    if (smin < 0.0f) smin = 1.0f; if (smax < smin) smax = smin;
    float s = smin + (smax - smin) * q;
    if (s < smin) s = smin; if (s > smax) s = smax;
    return s;
}

/* R03 CATTS contrastive adaptive token scaling: allowed = round(draft * c),
 * clamped to [1, draft]. Low confidence c -> fewer tokens. */
int wubu_catts_tokens(int draft_len, float conf) {
    if (draft_len <= 0) return 0;
    if (conf < 0.0f) conf = 0.0f; if (conf > 1.0f) conf = 1.0f;
    int t = (int)((float)draft_len * conf + 0.5f);
    if (t < 1) t = 1; if (t > draft_len) t = draft_len;
    return t;
}
