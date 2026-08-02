/*
 * wubu_agentic_os.c -- AGI-OS agentic runtime governance (AD01-AD04). C11.
 *
 * Convergence (AgentCgroup 2026 / 9P capability surface / durable-exec 7-hop):
 *   - AD01 9P capability enforcement: each agent gets a bounded subtree of the
 *          Styx namespace (a prefix path); access outside its subtree is denied.
 *   - AD02 skip-if-running + exponential backoff scheduler (cron-style).
 *   - AD03 durable-execution resume: state checkpoint [seq,step] survives restart.
 *   - AD04 cgroup-style resource bound: declare cpu_ms/ram_mb/io_budget; overrun
 *          flagged (no real cgroup syscall -- pure policy, portable C11).
 *
 * All pure C11, no third-party deps. Deterministic, testable.
 */
#include "wubu_agentic_os.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* AD01: 9P capability check -- path must live under agent_subtree. */
int wubu_9p_cap_allowed(const char *agent_subtree, const char *path) {
    if (!agent_subtree || !path) return 0;
    size_t sl = strlen(agent_subtree);
    if (strncmp(path, agent_subtree, sl) != 0) return 0;
    /* exact match or child (path[sl] must be '/' or '\0') */
    if (path[sl] == '/' || path[sl] == '\0') return 1;
    return 0;
}

/* AD02: exponential backoff. attempt>=0 -> delay_ms = base * 2^min(attempt,cap). */
long wubu_backoff_ms(int attempt, long base, int cap) {
    if (attempt < 0) attempt = 0;
    if (base < 1) base = 1;
    if (cap < 0) cap = 0;
    int e = attempt < cap ? attempt : cap;
    long d = base;
    for (int i = 0; i < e; i++) d *= 2;   /* portable 2^e */
    return d;
}

/* AD02: skip-if-running predicate. */
int wubu_skip_if_running(int running_flag) { return running_flag ? 1 : 0; }

/* AD03: pack/resume checkpoint. */
void wubu_checkpoint_pack(wubu_checkpoint_t *c, long seq, int step) {
    if (!c) return;
    c->seq = seq; c->step = step;
}
int wubu_checkpoint_resume(const wubu_checkpoint_t *c, long *seq, int *step) {
    if (!c || !seq || !step) return 0;
    *seq = c->seq; *step = c->step;
    return 1;
}

/* AD04: resource budget check. overrun -> returns bitmask of which bounds exceeded. */
int wubu_resbound_check(const wubu_resbound_t *b, long cpu_ms, long ram_mb, long io_kb) {
    if (!b) return 0;
    int over = 0;
    if (cpu_ms > b->cpu_ms_max) over |= 1;
    if (ram_mb > b->ram_mb_max) over |= 2;
    if (io_kb  > b->io_kb_max)  over |= 4;
    return over;   /* 0 = within budget */
}
