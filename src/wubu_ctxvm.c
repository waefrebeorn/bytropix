/*
 * wubu_ctxvm.c -- AGI-OS context virtual-memory hierarchy (AF08-AF10). C11.
 *
 * Convergence (LLM context as virtual memory / demand paging / 4-level 7-hop):
 *   - AF08 4-level context hierarchy: L1 gen window, L2 session, L3 long-term,
 *          L4 cross-session. Classify a memory by tier + promote/demote.
 *   - AF09 demand-paging eviction: FIFO + working-set. Given a KV ring of fixed
 *          capacity, evict oldest (FIFO) when over capacity; working-set keeps
 *          recent window resident.
 *   - AF10 semantic cache reuse across agents: two prompts within cosine sim
 *          threshold share a cached result (vector similarity).
 *
 * Pure C11, deterministic, testable.
 */
#include "wubu_ctxvm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* AF08 tier classify by importance + ttl (mirrors agentic_mem tiers). */
int wubu_ctx_tier(float importance, long ttl) {
    if (importance >= 0.8f && ttl > 1000) return WUBU_CTX_L4;
    if (importance >= 0.4f && ttl > 10)   return WUBU_CTX_L3;
    if (ttl > 0)                          return WUBU_CTX_L2;
    return WUBU_CTX_L1;
}

/* AF09 FIFO demand-paging evictor over a fixed-capacity KV ring.
 * Call per token: push token id; if size>capacity, evict oldest (FIFO).
 * Returns 1 if an eviction occurred this call, else 0. */
int wubu_ctx_evict_fifo(wubu_ctxring_t *r, long tok) {
    if (!r) return 0;
    int evicted = 0;
    if (r->size >= r->capacity) {
        /* evict oldest: shift head */
        if (r->size > 0) { r->head = (r->head + 1) % r->capacity; r->size--; evicted = 1; }
    }
    int pos = (r->head + r->size) % r->capacity;
    r->tok[pos] = tok;
    r->size++;
    return evicted;
}

/* working-set resident check: token within last `ws` positions is resident. */
int wubu_ctx_resident(const wubu_ctxring_t *r, long tok, int ws) {
    if (!r || ws <= 0) return 0;
    for (int i = 0; i < r->size && i < ws; i++) {
        int idx = (r->head + r->size - 1 - i + r->capacity) % r->capacity;
        if (r->tok[idx] == tok) return 1;
    }
    return 0;
}

/* AF10 cosine similarity of two equal-length vectors (for semantic cache). */
float wubu_cosine(const float *a, const float *b, int n) {
    if (!a || !b || n <= 0) return 0.0f;
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    if (na <= 0 || nb <= 0) return 0.0f;
    return (float)(dot / (sqrt(na) * sqrt(nb)));
}

/* AF10: share cache if cosine >= thr. */
int wubu_sem_cache_hit(const float *q, const float *cached, int n, float thr) {
    return wubu_cosine(q, cached, n) >= thr ? 1 : 0;
}
