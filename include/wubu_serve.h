/*
 * wubu_serve.h -- multi-tenant serving scheduler (Theme IR). C11.
 * Token-budget admission, fair-share, preemption with cost models,
 * LCP scheduling, decoupled scheduling, burst handling, priority
 * tiers with starvation bounds, tenant isolation, victim selection,
 * checkpointed preemption, SLO-awareness, batch compaction, hysteresis.
 */
#ifndef WUBU_SERVE_H
#define WUBU_SERVE_H

#include <stdint.h>

/* IR01: admission control by the token budget. */
int wubu_serve_admit(long used_tokens, long budget, long req_tokens);

/* IR02: fair-share fraction (weighted). */
float wubu_serve_fair_share(long tokens, long total, float weight);

/* IR03: preempt-vs-restart decision by cost. */
int wubu_serve_preempt(float rebuild_cost, float wait_cost);

/* IR04: activation-budget preemption guard. */
int wubu_serve_guard(long used_mem, long threshold, long req_mem);

/* IR06: longest-common-prefix length (shared prefill savings). */
int wubu_serve_lcp(const uint32_t *a, const uint32_t *b, int n);

/* IR07: decoupled scheduling (decision + resource split). */
typedef struct { int decision; long reserved; } wubu_serve_dec_t;
int wubu_serve_decouple(int schedule_ok, long available, long need,
                        wubu_serve_dec_t *out);

/* IR09: burst handling -- elastic admission headroom. */
long wubu_serve_burst_headroom(long steady, float burst_factor, long budget);

/* IR10: priority tier admission with starvation bounds. */
int wubu_serve_tier_admit(int tier, long *starvation, long bound);

/* IR11: per-tenant partition size. */
long wubu_serve_tenant_share(long total, int n_tenants);

/* IR13: victim selection -- the cheapest-to-restart. */
int wubu_serve_victim(const float *rebuild_cost, int n);

/* IR14: checkpointed preemption (snapshot cost vs restart cost). */
int wubu_serve_checkpoint(float snapshot_cost, float restart_cost);

/* IR15: SLO-aware scheduling (deadline slack). */
float wubu_serve_slo_slack(float deadline, float now, float eta);

/* IR16: batch compaction -- fill decode gaps with prefill chunks. */
int wubu_serve_compact(long decode_slots, long prefill_chunks, long *fill);

/* IR19: priority inheritance for requests. */
int wubu_serve_pi(int requester_prio, int holder_prio, int *inherited);

/* IR20: hysteresis -- avoid accept/preempt oscillation. */
int wubu_serve_hysteresis(long mem, long hi, long lo, int *state);

#endif
