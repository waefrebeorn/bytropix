/*
 * wubu_serve2.h -- the serving-scheduler frontier, complete (IR). C11.
 * Agnostic + data-driven: a scheduler-state struct + policy-selector
 * table, so the caller picks the mechanism instead of the module
 * hardcoding it. Covers fairness monitors, telemetry, co-scheduling,
 * cache-aware routing, budget profiling, work-conserving, backfill,
 * arbitration, cost-awareness, resilience, isolation, coalescing,
 * deadlines, benchmarking, the operator.
 */
#ifndef WUBU_SERVE2_H
#define WUBU_SERVE2_H

#include <stdint.h>

/* IR21: fairness metric (the achieved share vs the entitled share). */
float wubu_serve2_fairness(long achieved, long entitled);

/* IR22: preemption telemetry (rate + cost). */
float wubu_serve2_ptel(long preempts, long total, float avg_cost);

/* IR23: co-scheduling prefill+decode (the interleave ratio). */
int wubu_serve2_cosched(long prefill, long decode, float ratio, long *fill);

/* IR24: cache-aware routing (the node with the shared prefix). */
int wubu_serve2_route(const int *prefix_len, int n_nodes);

/* IR25: token-demand profiler (the request's estimated growth). */
long wubu_serve2_prof(long prompt_len, float growth_rate);

/* IR26: admission by predicted KV growth. */
int wubu_serve2_admit_pred(long used, long budget, long predicted);

/* IR27: work-conserving (never idle with queued work). */
int wubu_serve2_work_conserving(long queued, long running, long capacity);

/* IR28: per-tenant preemption budget. */
int wubu_serve2_pbudget(long used, long budget);

/* IR29: decode priority under contention. */
int wubu_serve2_decode_prio(int is_decode, long contention);

/* IR31: multi-queue (per-SLO-class queues). */
int wubu_serve2_queue(long *queue_loads, int nq, int *chosen);

/* IR32: backfill with background work. */
long wubu_serve2_backfill(long idle_slots, long bg_ready);

/* IR34: context-keepalive (hot contexts stay resident). */
int wubu_serve2_keepalive(float hotness, float th);

/* IR35: evict-vs-preempt arbitration. */
int wubu_serve2_arbitrate(float evict_cost, float preempt_cost);

/* IR36: cost-aware scheduling (J/token). */
float wubu_serve2_cost(long tokens, float j_per_token);

/* IR37: burst-adaptive fairness weights. */
float wubu_serve2_burst_weight(float demand, float steady);

/* IR38: prefix-similarity grouping. */
int wubu_serve2_group(const uint32_t *a, const uint32_t *b, int n, int th);

/* IR39: preemption-recovery speedup (checkpoint restore). */
float wubu_serve2_restore(float snapshot, float full);

/* IR40: scheduler resilience (no request loss on restart). */
int wubu_serve2_resilient(long in_flight, long recovered);

/* IR41: tenant isolation boundaries. */
int wubu_serve2_isolate(long tenant_used, long tenant_cap);

/* IR42: token-budget debt. */
int wubu_serve2_debt(long overspend, long *debt, long grace);

/* IR43: SLO violation monitor. */
int wubu_serve2_slo_violation(float deadline, float actual, float slack);

/* IR44: adaptive concurrency (in-flight by memory pressure). */
long wubu_serve2_concurrency(long mem_free, long mem_total, long max_inflight);

/* IR45: the policy selector. */
int wubu_serve2_policy_select(float load, float burstiness);

/* IR46: idle-capacity scavenging. */
int wubu_serve2_scavenge(float idle_frac, float th);

/* IR47: request coalescing. */
int wubu_serve2_coalesce(const uint32_t *a, const uint32_t *b, int n,
                         float sim_th);

/* IR48: preemption cost-benefit. */
int wubu_serve2_cost_benefit(float restart, float preempt, float save);

/* IR49: memory-pressure feedback. */
int wubu_serve2_feedback(float pressure, float th, long *concurrency);

/* IR50: deadline-aware scheduling. */
int wubu_serve2_deadline(float deadline, float now, float eta);

/* IR51: fair preemption order (least-SLO-critical first). */
int wubu_serve2_fair_preempt(const float *slo_criticality, int n);

/* IR53: shared-prefix cache with accounting. */
long wubu_serve2_shared_save(const int *prefix_lens, int n, long per_token);

/* IR55: multi-model scheduling. */
int wubu_serve2_multi_model(long *model_loads, int n_models, int *chosen);

/* IR56: scheduler hysteresis. */
int wubu_serve2_sched_hysteresis(float load, float hi, float lo, int *state);

/* IR57: queue-depth telemetry. */
float wubu_serve2_qdepth(const long *depths, int nq);

/* IR58: request aging. */
int wubu_serve2_aging(long age, long max_age, float *prio);

/* IR60: preemption simulation (dry-run). */
int wubu_serve2_simulate(const float *costs, int n, float budget, int *victims,
                         int cap);

/* IR62: budget negotiation. */
int wubu_serve2_negotiate(long used, long cap, long request);

/* IR63: memory-debt reclamation. */
long wubu_serve2_reclaim(long overspend, long rate);

/* IR64: prefill batch planning. */
int wubu_serve2_prefill_plan(long total_tokens, long chunk, long *n_chunks);

/* IR65: the schedule event log. */
int wubu_serve2_log(uint32_t *log, int n, uint32_t entry);

/* IR67: the power-cap envelope. */
int wubu_serve2_powercap(long tokens, float jpt, float power_budget);

#endif
