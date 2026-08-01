/*
 * wubu_integrate.c -- Runtime policy composer wiring the recursive-loop gap
 * modules into the live decode path (option c: exploit discovered gaps).
 *
 * Convergence (capacity_wall + kv_budget + stream_kv + ctx_manage + sys_tune +
 * linear_attn hybrid + pd_serve MoD 7-hop): this is the glue the loop's operator
 * needs. Each decode step calls wubu_decode_policy_step(), which:
 *   1. capacity_wall  -- guard: if next token would exceed the 512K OOM ceiling,
 *                         force eviction / stream window (never EAMM).
 *   2. kv_budget      -- compute per-layer keep-budget for the current ctx len.
 *   3. stream_kv      -- decide sink+window eviction if over budget.
 *   4. ctx_manage     -- elastic advice (which tokens to evict) under pressure.
 *   5. hybrid         -- for linear-attn hybrid layers, decide recurrent vs attn.
 *   6. mod/pd         -- per-token MoD execute + (if PD enabled) pull-route.
 *
 * It is PURE POLICY: it never allocates KV, it returns decisions the engine
 * applies. This keeps the integration safe (no rewrite of wubu_model.c) and
 * testable. The operator (recursive_optimize) feeds env-overridable params.
 *
 * Triple-DA: null ctx -> no-op; dims clamped; deterministic.
 */
#include "wubu_integrate.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct wubu_decode_policy {
    int max_ctx;          /* 512K ceiling (EAMM guard) */
    int n_layers;
    int hybrid_period;    /* 0 = no hybrid (all attention) */
    int pd_enabled;
    int sink;             /* stream_kv sink */
    int window;           /* stream_kv rolling window */
    float base_budget;    /* kv_budget base keep fraction */
    float block_low;      /* ctx_manage low-information block floor */
};

wubu_decode_policy_t *wubu_decode_policy_create(int max_ctx, int n_layers) {
    if (max_ctx <= 0 || n_layers <= 0) return NULL;
    wubu_decode_policy_t *p = (wubu_decode_policy_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->max_ctx = max_ctx;
    p->n_layers = n_layers;
    p->hybrid_period = 0;      /* default: dense attention */
    p->pd_enabled = 0;
    p->sink = 4;
    p->window = 512;
    p->base_budget = 1.0f;
    p->block_low = 0.02f;
    return p;
}

void wubu_decode_policy_destroy(wubu_decode_policy_t *p) { free(p); }

void wubu_decode_policy_set_hybrid(wubu_decode_policy_t *p, int period) {
    if (p && period > 0) p->hybrid_period = period;
}
void wubu_decode_policy_set_pd(wubu_decode_policy_t *p, int on) {
    if (p) p->pd_enabled = on ? 1 : 0;
}

/*
 * Per-step decision. Fills `out`:
 *   out->oom_safe      : 1 if next token fits under 512K (else eviction forced)
 *   out->force_evict   : 1 if over window budget -> stream_kv eviction advised
 *   out->keep_budget   : per-layer keep fraction (kv_budget mean)
 *   out->elastic_evict : tokens to evict (ctx_manage) if under pressure
 *   out->hybrid_recurrent : 1 if current layer L is recurrent (linear_attn)
 *   out->pd_accept     : 1 if decode can pull (pd_serve pull-route)
 */
void wubu_decode_policy_step(const wubu_decode_policy_t *p, int seqlen,
                             int decode_qlen, int high_water, int L,
                             wubu_decode_decision_t *out) {
    if (!p || !out) { if (out) memset(out, 0, sizeof(*out)); return; }
    memset(out, 0, sizeof(*out));

    /* 1. capacity_wall: never exceed max_ctx (EAMM guard). */
    int next = seqlen + 1;
    out->oom_safe = (next <= p->max_ctx) ? 1 : 0;
    if (!out->oom_safe) out->force_evict = 1; /* stream to recover headroom */

    /* 2. stream_kv eviction if over window budget. */
    if (seqlen > p->sink + p->window) out->force_evict = 1;

    /* 3. kv_budget: keep fraction. Deeper ctx -> tighter (mean = base_budget). */
    float frac = p->base_budget;
    if (seqlen > p->max_ctx / 2) frac *= 0.9f; /* mild tighten past half */
    if (frac < 0.1f) frac = 0.1f;
    out->keep_budget = frac;

    /* 4. ctx_manage: elastic eviction count under pressure (sliding tail). */
    out->elastic_evict = out->force_evict ? (seqlen - (p->sink + p->window)) : 0;
    if (out->elastic_evict < 0) out->elastic_evict = 0;

    /* 5. hybrid (linear_attn): recurrent unless layer is the attention boundary. */
    if (p->hybrid_period > 0)
        out->hybrid_recurrent = (L % p->hybrid_period == 0) ? 0 : 1;

    /* 6. pd_serve pull-route. */
    if (p->pd_enabled)
        out->pd_accept = (decode_qlen < high_water) ? 1 : 0;
}

/* Convenience: env-driven global policy (operator feeds WUBU_* overrides). */
wubu_decode_policy_t *wubu_decode_policy_default(int max_ctx, int n_layers) {
    wubu_decode_policy_t *p = wubu_decode_policy_create(max_ctx, n_layers);
    if (!p) return NULL;
    const char *e;
    if ((e = getenv("WUBU_HYBRID_PERIOD"))) p->hybrid_period = atoi(e);
    if ((e = getenv("WUBU_PD"))) p->pd_enabled = atoi(e) ? 1 : 0;
    if ((e = getenv("WUBU_STREAM_SINK"))) p->sink = atoi(e);
    if ((e = getenv("WUBU_STREAM_WINDOW"))) p->window = atoi(e);
    if ((e = getenv("WUBU_KV_BUDGET"))) { float f = (float)atof(e); if (f>0) p->base_budget = f; }
    return p;
}
