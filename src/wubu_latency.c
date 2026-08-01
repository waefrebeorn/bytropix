/*
 * wubu_latency.c -- AGI-OS latency-class scheduler (AF05-AF07). C11.
 *
 * Convergence (Agent-OS HRT/SRT/DT / EDF-RM / Agent-Contract SLO 7-hop):
 *   - AF05 latency class (HRT/SRT/DT) + EDF/RM-ready scheduler hook: earliest-
 *          deadline-first ordering of agent tasks; returns the run order.
 *   - AF06 WCET + jitter budget accounting: track worst-case exec time and the
 *          jitter (variation) vs the declared budget; flag misses.
 *   - AF07 Agent-Contract SLO enforcement: TTFT / full-turn / throughput SLOs
 *          checked against measured values; returns pass/fail per class.
 *
 * Pure C11, deterministic, testable.
 */
#include "wubu_latency.h"
#include <stdlib.h>
#include <string.h>

/* AF05: EDF sort of task deadlines (ascending). Returns 1 if reordered. */
int wubu_edf_order(wubu_task_t *t, int n) {
    if (!t || n < 2) return 0;
    /* insertion sort by deadline (small arrays; deterministic, no libc qsort dep) */
    int reordered = 0;
    for (int i = 1; i < n; i++) {
        wubu_task_t key = t[i];
        int j = i - 1;
        while (j >= 0 && t[j].deadline_ms > key.deadline_ms) {
            t[j + 1] = t[j]; j--;
        }
        if (j + 1 != i) reordered = 1;
        t[j + 1] = key;
    }
    return reordered;
}

/* AF06: WCET + jitter accounting for one agent's samples. */
void wubu_wcet_account(const long *samples, int n, wubu_wcet_t *out) {
    if (!out) return;
    out->wcet_ms = 0; out->mean_ms = 0; out->jitter_ms = 0;
    if (!samples || n <= 0) return;
    long sum = 0, mx = samples[0];
    for (int i = 0; i < n; i++) {
        sum += samples[i];
        if (samples[i] > mx) mx = samples[i];
    }
    out->wcet_ms = mx;
    out->mean_ms = (double)sum / (double)n;
    /* jitter = max deviation from mean */
    double jit = 0;
    for (int i = 0; i < n; i++) {
        double d = samples[i] > out->mean_ms ? samples[i] - out->mean_ms
                                             : out->mean_ms - samples[i];
        if (d > jit) jit = d;
    }
    out->jitter_ms = jit;
}

/* AF06: deadline-miss flag: wcet exceeds the class budget. */
int wubu_deadline_miss(const wubu_wcet_t *w, long budget_ms) {
    return (w && w->wcet_ms > budget_ms) ? 1 : 0;
}

/* AF07: Agent-Contract SLO check. Returns bitmask of failed dimensions. */
int wubu_slo_check(wubu_latclass_t cls, const wubu_slo_meas_t *m) {
    if (!m) return 0;
    int fail = 0;
    if (cls == WUBU_LC_HRT) {
        if (m->ttft_ms  > 20)  fail |= 1;   /* onset <=20ms */
        if (m->turn_ms  > 20)  fail |= 2;   /* full-turn slice <=20ms */
        if (m->jitter_ms > 5)  fail |= 4;   /* jitter <=5ms */
    } else if (cls == WUBU_LC_SRT) {
        if (m->ttft_ms  > 300) fail |= 1;   /* TTFT 150-300ms */
        if (m->turn_ms  > 1200) fail |= 2;  /* full-turn 0.8-1.2s */
        if (m->jitter_ms > m->turn_ms * 0.20) fail |= 4; /* P95 <=20% */
    } else { /* DT: throughput first */
        if (m->tok_per_sec < 1.0) fail |= 8; /* must make progress */
    }
    return fail; /* 0 = all SLOs met */
}
