/*
 * wubu_agentic_mem.c -- AGI-OS agentic memory (AE01-AE04). C11.
 *
 * Convergence (TeleMem/HiMem/Redis 2026 3-tier + consolidation 7-hop):
 *   - AE01 episodic->semantic consolidation: an episodic event is "distillable"
 *          when its importance exceeds a threshold (raw episode can then drop).
 *   - AE02 semantic dedup: two facts with equal key collide -> keep higher-importance.
 *   - AE03 hierarchical tiers: classify a memory by its TTL/importance into
 *          working / session / long-term.
 *   - AE04 retrieval ranking: score = importance * recency_decay(t), with a
 *          forgetting curve so stale low-importance memories rank last.
 *
 * Pure C11, deterministic, testable.
 */
#include "wubu_agentic_mem.h"
#include <math.h>
#include <string.h>

/* AE03 tier classifier (working < session < long-term by importance+ttl). */
int wubu_mem_tier(float importance, long ttl_steps) {
    if (importance >= 0.8f && ttl_steps > 1000) return WUBU_TIER_LONGTERM;
    if (importance >= 0.4f && ttl_steps > 10)   return WUBU_TIER_SESSION;
    return WUBU_TIER_WORKING;
}

/* AE01 consolidation predicate: distill when importance >= threshold. */
int wubu_mem_consolidate(float importance, float thresh) {
    return (importance >= thresh) ? 1 : 0;
}

/* AE02 dedup: given existing fact importance and new one, return which to keep
 * (1 = keep existing, 2 = keep new, 0 = equal). */
int wubu_mem_dedup(float imp_existing, float imp_new) {
    if (imp_new > imp_existing) return 2;
    if (imp_existing > imp_new) return 1;
    return 0;
}

/* AE04 retrieval score with forgetting curve: imp * exp(-age/tau). */
float wubu_mem_retrieval_score(float importance, long age, long tau) {
    if (tau <= 0) tau = 1;
    if (age < 0) age = 0;
    return importance * (float)expf(-(float)age / (float)tau);
}

/* AE01 helper: key-based equal (for dedup hit test). */
int wubu_mem_key_eq(const char *a, const char *b) {
    if (!a || !b) return 0;
    return strcmp(a, b) == 0 ? 1 : 0;
}
