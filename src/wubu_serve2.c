/*
 * wubu_serve2.c -- the serving-scheduler frontier, complete (IR). C11.
 */
#include "wubu_serve2.h"

float wubu_serve2_fairness(long achieved, long entitled)
{
    if (entitled <= 0) return 0;
    return (float)achieved / (float)entitled;
}

float wubu_serve2_ptel(long preempts, long total, float avg_cost)
{
    if (total <= 0) return 0;
    return ((float)preempts / (float)total) * avg_cost;
}

int wubu_serve2_cosched(long prefill, long decode, float ratio, long *fill)
{
    if (!fill || ratio <= 0) return -1;
    long target = (long)((float)decode / ratio);
    *fill = prefill < target ? prefill : target;
    return 0;
}

int wubu_serve2_route(const int *prefix_len, int n_nodes)
{
    if (!prefix_len || n_nodes <= 0) return -1;
    int best = 0;
    for (int i = 1; i < n_nodes; i++)
        if (prefix_len[i] > prefix_len[best]) best = i;
    return best;
}

long wubu_serve2_prof(long prompt_len, float growth_rate)
{
    return prompt_len + (long)((float)prompt_len * growth_rate);
}

int wubu_serve2_admit_pred(long used, long budget, long predicted)
{
    if (budget <= 0) return 0;
    return (used + predicted <= budget) ? 1 : 0;
}

int wubu_serve2_work_conserving(long queued, long running, long capacity)
{
    return (queued > 0 && running < capacity) ? 1 : 0;
}

int wubu_serve2_pbudget(long used, long budget)
{
    if (budget <= 0) return 0;
    return used <= budget ? 1 : 0;
}

int wubu_serve2_decode_prio(int is_decode, long contention)
{
    return (is_decode && contention > 0) ? 1 : 0;
}

int wubu_serve2_queue(long *queue_loads, int nq, int *chosen)
{
    if (!queue_loads || !chosen || nq <= 0) return -1;
    int best = 0;
    for (int i = 1; i < nq; i++)
        if (queue_loads[i] < queue_loads[best]) best = i;
    *chosen = best;
    return 0;
}

long wubu_serve2_backfill(long idle_slots, long bg_ready)
{
    return idle_slots < bg_ready ? idle_slots : bg_ready;
}

int wubu_serve2_keepalive(float hotness, float th)
{
    return hotness >= th ? 1 : 0;
}

int wubu_serve2_arbitrate(float evict_cost, float preempt_cost)
{
    return evict_cost < preempt_cost ? 1 : 0;
}

float wubu_serve2_cost(long tokens, float j_per_token)
{
    return (float)tokens * j_per_token;
}

float wubu_serve2_burst_weight(float demand, float steady)
{
    if (steady <= 0) return 1.0f;
    return demand / steady;
}

int wubu_serve2_group(const uint32_t *a, const uint32_t *b, int n, int th)
{
    if (!a || !b) return -1;
    int same = 0;
    while (same < n && a[same] == b[same]) same++;
    return same >= th ? 1 : 0;
}

float wubu_serve2_restore(float snapshot, float full)
{
    return snapshot / (full > 0 ? full : 1.0f);
}

int wubu_serve2_resilient(long in_flight, long recovered)
{
    return recovered >= in_flight ? 1 : 0;
}

int wubu_serve2_isolate(long tenant_used, long tenant_cap)
{
    if (tenant_cap <= 0) return 0;
    return tenant_used <= tenant_cap ? 1 : 0;
}

int wubu_serve2_debt(long overspend, long *debt, long grace)
{
    if (!debt) return -1;
    if (overspend > grace) { *debt += overspend - grace; return 1; }
    return 0;
}

int wubu_serve2_slo_violation(float deadline, float actual, float slack)
{
    return actual > deadline + slack ? 1 : 0;
}

long wubu_serve2_concurrency(long mem_free, long mem_total, long max_inflight)
{
    if (mem_total <= 0) return max_inflight;
    float pressure = (float)(mem_total - mem_free) / (float)mem_total;
    long cap = (long)((float)max_inflight * (1.0f - pressure));
    return cap < 1 ? 1 : cap;
}

int wubu_serve2_policy_select(float load, float burstiness)
{
    if (burstiness > 0.7f) return 2;   /* burst-adaptive */
    if (load > 0.7f) return 1;         /* throughput */
    return 0;                          /* balanced */
}

int wubu_serve2_scavenge(float idle_frac, float th)
{
    return idle_frac >= th ? 1 : 0;
}

int wubu_serve2_coalesce(const uint32_t *a, const uint32_t *b, int n,
                         float sim_th)
{
    if (!a || !b || n <= 0) return -1;
    int same = 0;
    while (same < n && a[same] == b[same]) same++;
    float sim = (float)same / (float)n;
    return sim >= sim_th ? 1 : 0;
}

int wubu_serve2_cost_benefit(float restart, float preempt, float save)
{
    return (restart - preempt) > save ? 1 : 0;
}

int wubu_serve2_feedback(float pressure, float th, long *concurrency)
{
    if (!concurrency) return -1;
    if (pressure > th) { *concurrency = *concurrency > 1 ? *concurrency / 2 : 1; return 1; }
    *concurrency *= 2;
    return 0;
}

int wubu_serve2_deadline(float deadline, float now, float eta)
{
    return (now + eta) <= deadline ? 1 : 0;
}

int wubu_serve2_fair_preempt(const float *slo_criticality, int n)
{
    if (!slo_criticality || n <= 0) return -1;
    int best = 0;
    for (int i = 1; i < n; i++)
        if (slo_criticality[i] < slo_criticality[best]) best = i;
    return best;
}

long wubu_serve2_shared_save(const int *prefix_lens, int n, long per_token)
{
    if (!prefix_lens) return 0;
    long best = 0;
    for (int i = 0; i < n; i++)
        if (prefix_lens[i] > best) best = prefix_lens[i];
    return best * per_token;
}

int wubu_serve2_multi_model(long *model_loads, int n_models, int *chosen)
{
    if (!model_loads || !chosen || n_models <= 0) return -1;
    int best = 0;
    for (int i = 1; i < n_models; i++)
        if (model_loads[i] < model_loads[best]) best = i;
    *chosen = best;
    return 0;
}

int wubu_serve2_sched_hysteresis(float load, float hi, float lo, int *state)
{
    if (!state) return -1;
    if (load >= hi) *state = 1;
    else if (load <= lo) *state = 0;
    return *state;
}

float wubu_serve2_qdepth(const long *depths, int nq)
{
    if (!depths || nq <= 0) return 0;
    long s = 0;
    for (int i = 0; i < nq; i++) s += depths[i];
    return (float)s / (float)nq;
}

int wubu_serve2_aging(long age, long max_age, float *prio)
{
    if (!prio) return -1;
    *prio = (float)age / (float)(max_age > 0 ? max_age : 1);
    return age >= max_age ? 1 : 0;   /* force-admit when starved */
}

int wubu_serve2_simulate(const float *costs, int n, float budget, int *victims,
                         int cap)
{
    if (!costs || !victims || cap <= 0) return -1;
    float spent = 0;
    int k = 0;
    for (int i = 0; i < n && k < cap; i++) {
        if (spent + costs[i] <= budget) { victims[k++] = i; spent += costs[i]; }
    }
    return k;
}

int wubu_serve2_negotiate(long used, long cap, long request)
{
    if (cap <= 0) return 0;
    return (used + request <= cap * 2) ? 1 : 0;   /* 2x burst headroom */
}

long wubu_serve2_reclaim(long overspend, long rate)
{
    return overspend < rate ? 0 : overspend - rate;
}

int wubu_serve2_prefill_plan(long total_tokens, long chunk, long *n_chunks)
{
    if (!n_chunks || chunk <= 0) return -1;
    *n_chunks = (total_tokens + chunk - 1) / chunk;
    return 0;
}

int wubu_serve2_log(uint32_t *log, int n, uint32_t entry)
{
    if (!log || n <= 0) return -1;
    for (int i = n - 1; i > 0; i--) log[i] = log[i - 1];
    log[0] = entry;
    return 0;
}

int wubu_serve2_powercap(long tokens, float jpt, float power_budget)
{
    if (power_budget <= 0) return 0;
    return (float)tokens * jpt <= power_budget ? 1 : 0;
}
