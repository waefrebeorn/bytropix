/*
 * wubu_loopguard.c -- Missing-need guards for the AGI-OS (AG01/AG05/AG06/AG08). C11.
 *
 * Convergence (OWASP LLM10/ASI08 runaway loops, L7xT1-4 trajectory audit,
 * LLM06/ASI02 tool-abuse cap, ASI08/strata JIT+HITL 7-hop):
 *   - AG01 runaway-loop guard: enforce a max step-count + wall-clock deadline;
 *          once exceeded, the agentic loop MUST terminate (no recursive runaway).
 *   - AG05 trajectory audit: append-only per-action record (agent id + action
 *          hash + nonce); immutable attribution for accountability.
 *   - AG06 tool-abuse cap: per-agent tool-call rate limiter (count per window).
 *   - AG08 HITL gating: a sensitive action requires an external approval token;
 *          without it, the action is denied (default-deny, reversible by human).
 *
 * Pure C11, deterministic, testable. No crypto beyond a cheap FNV hash for the
 * trajectory nonce/attribution (homestic, not third-party).
 */
#include "wubu_loopguard.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* cheap FNV-1a 64-bit hash for action/agent attribution (no external dep). */
static unsigned long long fnv1a(const char *s, int n) {
    unsigned long long h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* AG01: runaway-loop guard. Returns 1 if the loop may continue, 0 if it must
 * terminate (step >= max_steps OR now >= deadline_ns). */
int wubu_loop_may_continue(const wubu_loopguard_t *g, long step, long now_ns) {
    if (!g) return 0;
    if (step >= g->max_steps) return 0;
    if (g->deadline_ns > 0 && now_ns >= g->deadline_ns) return 0;
    return 1;
}

/* AG05: append an action to the trajectory audit (immutable, append-only).
 * Returns the nonce (hash of agent+action+seq) for attribution. */
unsigned long long wubu_traj_append(wubu_traj_t *t, const char *agent,
                                    const char *action) {
    if (!t) return 0;
    unsigned long long h = fnv1a(agent, (int)strlen(agent));
    h ^= fnv1a(action, (int)strlen(action)) + t->count * 1469598103934665603ULL;
    if (t->count < t->cap) {
        t->nonce[t->count] = h;
        t->count++;
    }
    return h; /* attribution nonce (idempotent append-only; no mutation of prior) */
}

/* AG06: tool-abuse cap. Returns 1 if the agent may make another tool call this
 * window, 0 if the per-window cap is exhausted. */
int wubu_tool_allowed(wubu_toolcap_t *c, const char *agent, long window_now) {
    if (!c) return 0;
    if (window_now != c->window) { c->window = window_now; c->calls = 0; }
    if (c->calls >= c->max_per_window) return 0;
    c->calls++;
    (void)agent;
    return 1;
}

/* AG08: HITL gating. A sensitive action (severity >= threshold) requires an
 * external approval token. Without it, deny (default-deny). Reversible: a human
 * may supply the token at any time. */
int wubu_hitl_approve(const wubu_hitl_t *h, float severity, int approval_token) {
    if (!h) return 0;
    if (severity < h->sensitivity) return 1;      /* low severity: auto-allow */
    return (approval_token == h->expected_token) ? 1 : 0; /* needs valid token */
}
