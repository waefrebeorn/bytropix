/*
 * wubu_integrate.h -- Runtime policy composer wiring the recursive-loop gap
 * modules into the live decode path (option c: exploit discovered gaps).
 *
 * This is the glue the operator (recursive_optimize) drives. Each decode step
 * calls wubu_decode_policy_step() to obtain OOM-safe / budget / eviction /
 * hybrid / PD decisions, which the engine applies. Pure policy -- no KV alloc.
 */
#ifndef WUBU_INTEGRATE_H
#define WUBU_INTEGRATE_H

#include <stddef.h>

typedef struct wubu_decode_policy wubu_decode_policy_t;

typedef struct {
    int oom_safe;          /* 1 if next token fits under 512K ceiling */
    int force_evict;       /* 1 if stream_kv eviction advised */
    float keep_budget;     /* per-layer keep fraction (kv_budget mean) */
    int elastic_evict;     /* tokens to evict (ctx_manage) under pressure */
    int hybrid_recurrent;  /* 1 if layer L is recurrent (linear_attn) */
    int pd_accept;         /* 1 if decode can pull (pd_serve) */
} wubu_decode_decision_t;

wubu_decode_policy_t *wubu_decode_policy_create(int max_ctx, int n_layers);
void wubu_decode_policy_destroy(wubu_decode_policy_t *p);
void wubu_decode_policy_set_hybrid(wubu_decode_policy_t *p, int period);
void wubu_decode_policy_set_pd(wubu_decode_policy_t *p, int on);

void wubu_decode_policy_step(const wubu_decode_policy_t *p, int seqlen,
                             int decode_qlen, int high_water, int L,
                             wubu_decode_decision_t *out);

/* Env-driven global policy (WUBU_* overrides). */
wubu_decode_policy_t *wubu_decode_policy_default(int max_ctx, int n_layers);

#endif /* WUBU_INTEGRATE_H */
