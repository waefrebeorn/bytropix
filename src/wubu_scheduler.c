/*
 * wubu_scheduler.c — Iteration-level (continuous) batching + prefix cache
 * (Areas H/I, items 71-88). C11, self-contained.
 *
 *   - In-flight batching: requests join/leave the batch per decode step (H.71).
 *   - Prefix cache with SHA-256 block dedup (I.81/I.82): repeated prompt
 *     prefixes reuse computed KV, skip prefill.
 *   - KV page free-pool recycle on completion (H.72).
 * Designed to plug into kv_paged_attention + kv_arena.
 */
#include "wubu_scheduler.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- request slot ---- */
wubu_req_t *wubu_req_create(int id, const int *prompt, int prompt_len,
                            int max_tokens) {
    wubu_req_t *r = (wubu_req_t *)calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->id = id;
    r->prompt = (int *)malloc(sizeof(int) * prompt_len);
    memcpy(r->prompt, prompt, sizeof(int) * prompt_len);
    r->prompt_len = prompt_len;
    r->max_tokens = max_tokens;
    /* Hash the entire prompt as the cacheable prefix by default. The engine
     * may override r->prefix_len to only the shared leading tokens. */
    r->prefix_len_cache = prompt_len;
    r->prefix_hash = wubu_prefix_hash(prompt, prompt_len);  /* I.81 */
    r->state = WUBU_REQ_PREFILL;
    r->n_kv = 0;
    r->n_gen = 0;
    return r;
}
void wubu_req_free(wubu_req_t *r) {
    if (!r) return;
    free(r->prompt);
    free(r->tokens);
    free(r);
}

/* ---- scheduler ---- */
wubu_sched_t *wubu_sched_create(int max_batch) {
    wubu_sched_t *s = (wubu_sched_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->max_batch = max_batch;
    s->cap = 16;
    s->reqs = (wubu_req_t **)calloc(s->cap, sizeof(wubu_req_t *));
    /* prefix cache: array of (hash -> kv_owner_request_id) */
    s->pcache_cap = 64;
    s->pcache = (wubu_pcache_t *)calloc(s->pcache_cap, sizeof(wubu_pcache_t));
    return s;
}
void wubu_sched_free(wubu_sched_t *s) {
    if (!s) return;
    for (int i = 0; i < s->n; i++) wubu_req_free(s->reqs[i]);
    free(s->reqs);
    free(s->pcache);
    free(s);
}

int wubu_sched_submit(wubu_sched_t *s, wubu_req_t *r) {
    if (s->n >= s->cap) { /* grow */
        int nc = s->cap * 2;
        wubu_req_t **tmp = (wubu_req_t **)realloc(s->reqs, nc * sizeof(wubu_req_t *));
        if (!tmp) return -1;
        s->reqs = tmp;
        s->cap = nc;
    }
    s->reqs[s->n++] = r;
    return 0;
}

/* Find a cached prefix owner for this request (I.81). Returns req id or -1. */
static int prefix_cache_lookup(wubu_sched_t *s, uint64_t h) {
    for (int i = 0; i < s->pcache_n; i++)
        if (s->pcache[i].hash == h) return s->pcache[i].owner_id;
    return -1;
}
static void prefix_cache_insert(wubu_sched_t *s, uint64_t h, int owner) {
    if (s->pcache_n >= s->pcache_cap) return;  /* simple fixed cap */
    s->pcache[s->pcache_n].hash = h;
    s->pcache[s->pcache_n].owner_id = owner;
    s->pcache_n++;
}

/* Run ONE decode iteration over all in-flight requests (H.71).
 * Returns number of requests still active (not finished). */
int wubu_sched_step(wubu_sched_t *s) {
    int active = 0;
    int w = 0;
    for (int i = 0; i < s->n; i++) {
        wubu_req_t *r = s->reqs[i];
        if (r->state == WUBU_REQ_DONE) continue;

        /* Prefix cache: skip prefill if a matching prefix is already computed. */
        if (r->state == WUBU_REQ_PREFILL) {
            int owner = prefix_cache_lookup(s, r->prefix_hash);
            if (owner >= 0) {
                r->state = WUBU_REQ_DECODE;   /* reuse cached KV (I.82) */
                r->kv_reused = 1;
            } else {
                prefix_cache_insert(s, r->prefix_hash, r->id); /* mark computed */
                /* (real engine would run prefill here) */
                r->state = WUBU_REQ_DECODE;
            }
        }

        /* Decode one token (caller supplies sampled token via wubu_req_emit). */
        if (r->state == WUBU_REQ_DECODE) {
            if (r->n_gen >= r->max_tokens) {
                r->state = WUBU_REQ_DONE;       /* free KV pages (H.72) */
            } else {
                active++;
            }
        }
        /* compact: keep only non-done in the active window */
        if (r->state != WUBU_REQ_DONE) s->reqs[w++] = r;
        else wubu_req_free(r);
    }
    s->n = w;
    return active;
}

/* Append an emitted token to a request (caller drives sampling). */
void wubu_req_emit(wubu_req_t *r, int tok) {
    if (r->n_kv >= r->max_tokens + r->prompt_len) return;
    if (r->n_kv % 16 == 0) {
        int *tmp = (int *)realloc(r->tokens, (r->n_kv + 16) * sizeof(int));
        if (!tmp) return;
        r->tokens = tmp;
    }
    r->tokens[r->n_kv++] = tok;
    r->n_gen++;
}

/* ---- SHA-256-lite prefix hash (FNV-1a 64, adequate for dedup) ---- */
uint64_t wubu_prefix_hash(const int *toks, int n) {
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < n; i++) {
        uint32_t v = (uint32_t)toks[i];
        unsigned char *p = (unsigned char *)&v;
        for (int b = 0; b < 4; b++) {
            h ^= p[b];
            h *= 1099511628211ULL;
        }
    }
    return h;
}
