/*
 * wubu_serve.c -- multi-tenant serving scheduler (Theme IR). C11.
 */
#include "wubu_serve.h"

int wubu_serve_admit(long used_tokens, long budget, long req_tokens)
{
    if (budget <= 0 || req_tokens < 0) return 0;
    return (used_tokens + req_tokens <= budget) ? 1 : 0;
}

float wubu_serve_fair_share(long tokens, long total, float weight)
{
    if (total <= 0) return 0;
    float wsum = 0;   /* the caller normalizes weights; here 1 tenant */
    (void)wsum;
    float share = (float)tokens / (float)total;
    return share * (weight > 0 ? weight : 1.0f);
}

int wubu_serve_preempt(float rebuild_cost, float wait_cost)
{
    /* preempt when rebuilding the cache is cheaper than waiting */
    return rebuild_cost < wait_cost ? 1 : 0;
}

int wubu_serve_guard(long used_mem, long threshold, long req_mem)
{
    if (threshold <= 0 || req_mem < 0) return 0;
    return (used_mem + req_mem <= threshold) ? 1 : 0;
}

int wubu_serve_lcp(const uint32_t *a, const uint32_t *b, int n)
{
    if (!a || !b) return 0;
    int k = 0;
    while (k < n && a[k] == b[k]) k++;
    return k;
}

int wubu_serve_decouple(int schedule_ok, long available, long need,
                        wubu_serve_dec_t *out)
{
    if (!out) return -1;
    out->decision = schedule_ok && (available >= need) ? 1 : 0;
    out->reserved = out->decision ? need : 0;
    return 0;
}

long wubu_serve_burst_headroom(long steady, float burst_factor, long budget)
{
    if (budget <= 0) return 0;
    long headroom = (long)((float)steady * (burst_factor - 1.0f));
    if (headroom < 0) headroom = 0;
    long cap = budget - steady;
    return headroom > cap ? cap : headroom;
}

int wubu_serve_tier_admit(int tier, long *starvation, long bound)
{
    if (!starvation) return 0;
    if (tier == 0) return 1;            /* highest tier always */
    (*starvation)++;
    if (*starvation >= bound) { *starvation = 0; return 1; }
    return 0;
}

long wubu_serve_tenant_share(long total, int n_tenants)
{
    if (n_tenants <= 0) return 0;
    return total / n_tenants;
}

int wubu_serve_victim(const float *rebuild_cost, int n)
{
    if (!rebuild_cost || n <= 0) return -1;
    int best = 0;
    for (int i = 1; i < n; i++)
        if (rebuild_cost[i] < rebuild_cost[best]) best = i;
    return best;
}

int wubu_serve_checkpoint(float snapshot_cost, float restart_cost)
{
    /* checkpoint when the snapshot is cheaper than a full restart */
    return snapshot_cost < restart_cost ? 1 : 0;
}

float wubu_serve_slo_slack(float deadline, float now, float eta)
{
    float slack = deadline - now - eta;
    return slack;
}

int wubu_serve_compact(long decode_slots, long prefill_chunks, long *fill)
{
    if (!fill) return -1;
    *fill = decode_slots < prefill_chunks ? decode_slots : prefill_chunks;
    return 0;
}

int wubu_serve_pi(int requester_prio, int holder_prio, int *inherited)
{
    if (!inherited) return -1;
    *inherited = requester_prio < holder_prio ? requester_prio : holder_prio;
    return 0;
}

int wubu_serve_hysteresis(long mem, long hi, long lo, int *state)
{
    if (!state) return -1;
    if (mem >= hi) *state = 1;       /* preempt zone */
    else if (mem <= lo) *state = 0;  /* accept zone */
    return *state;
}
