/*
 * wubu_agentic_mem.h -- AGI-OS agentic memory (AE01-AE04).
 */
#ifndef WUBU_AGENTIC_MEM_H
#define WUBU_AGENTIC_MEM_H

#define WUBU_TIER_WORKING   0
#define WUBU_TIER_SESSION   1
#define WUBU_TIER_LONGTERM  2

/* AE03 tier classifier. */
int  wubu_mem_tier(float importance, long ttl_steps);
/* AE01 consolidation predicate. */
int  wubu_mem_consolidate(float importance, float thresh);
/* AE02 dedup decision (1=keep existing, 2=keep new, 0=equal). */
int  wubu_mem_dedup(float imp_existing, float imp_new);
/* AE04 retrieval score with forgetting curve. */
float wubu_mem_retrieval_score(float importance, long age, long tau);
/* AE01/AE02 key equality. */
int  wubu_mem_key_eq(const char *a, const char *b);

#endif
