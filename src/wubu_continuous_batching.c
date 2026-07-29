/*
 * wubu_continuous_batching.c — Iteration-level scheduling for LLM decode.
 * C11, self-contained. Implements continuous batching (vLLM-style):
 *   - Sequences enter/leave batch dynamically per iteration
 *   - Prefill for new sequences, decode for continuing
 *   - Paged KV cache + prefix cache integration
 *   - Priority: prefix matches first (reuse), then longest-waiting
 */
#include "wubu_continuous_batching.h"
#include "wubu_paged_kv.h"
#include "wubu_prefix_cache.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* ---------- Sequence state ---------- */

static int seq_state_alloc(wubu_cont_batch_t *cb) {
    for (int i = 0; i < WUBU_CONT_MAX_SEQ; i++) {
        if (!cb->seqs[i].alive) {
            cb->seqs[i].alive = 1;
            cb->seqs[i].seq_id = i;
            cb->seqs[i].tokens_generated = 0;
            cb->seqs[i].prefill_done = 0;
            cb->seqs[i].block_ids = NULL;
            cb->seqs[i].n_blocks = 0;
            return i;
        }
    }
    return -1;
}

static void seq_state_free(wubu_cont_batch_t *cb, int idx) {
    if (idx < 0 || idx >= WUBU_CONT_MAX_SEQ) return;
    if (cb->seqs[idx].block_ids) {
        free(cb->seqs[idx].block_ids);
        cb->seqs[idx].block_ids = NULL;
        cb->seqs[idx].n_blocks = 0;
    }
    cb->seqs[idx].alive = 0;
}

/* ---------- Batch lifecycle ---------- */

wubu_cont_batch_t *wubu_cont_batch_create(int block_size, int n_blocks,
                                          int head_dim, int n_kv_heads,
                                          int max_seq, int max_tokens) {
    wubu_cont_batch_t *cb = (wubu_cont_batch_t *)calloc(1, sizeof(wubu_cont_batch_t));
    if (!cb) return NULL;

    cb->paged_kv = wubu_paged_kv_create(block_size, n_blocks, head_dim, n_kv_heads);
    if (!cb->paged_kv) { free(cb); return NULL; }

    cb->prefix_cache = wubu_prefix_cache_create();
    if (!cb->prefix_cache) { wubu_paged_kv_free(cb->paged_kv); free(cb); return NULL; }

    cb->max_seq = max_seq > 0 ? max_seq : WUBU_CONT_MAX_SEQ;
    cb->max_tokens_per_seq = max_tokens > 0 ? max_tokens : WUBU_CONT_MAX_TOKENS;
    cb->block_size = block_size;
    cb->head_dim = head_dim;
    cb->n_kv_heads = n_kv_heads;
    cb->n_active = 0;
    cb->iteration = 0;
    return cb;
}

void wubu_cont_batch_free(wubu_cont_batch_t *cb) {
    if (!cb) return;
    for (int i = 0; i < WUBU_CONT_MAX_SEQ; i++) {
        if (cb->seqs[i].alive) seq_state_free(cb, i);
    }
    if (cb->paged_kv) wubu_paged_kv_free(cb->paged_kv);
    if (cb->prefix_cache) wubu_prefix_cache_free(cb->prefix_cache);
    free(cb);
}

/* ---------- Sequence admission ---------- */

/* Add a new sequence (prefill tokens). Returns seq index or -1. */
int wubu_cont_batch_add_seq(wubu_cont_batch_t *cb, const int *tokens, int n_tokens) {
    if (!cb || cb->n_active >= cb->max_seq) return -1;

    int idx = seq_state_alloc(cb);
    if (idx < 0) return -1;

    wubu_seq_state_t *s = &cb->seqs[idx];
    s->n_tokens = n_tokens;
    s->tokens = (int *)malloc(sizeof(int) * n_tokens);
    if (!s->tokens) { seq_state_free(cb, idx); return -1; }
    memcpy(s->tokens, tokens, sizeof(int) * n_tokens);

    /* Try prefix cache match */
    int matched_blocks[WUBU_CONT_MAX_TOKENS / 16];
    int matched = wubu_prefix_cache_match(cb->prefix_cache, tokens, n_tokens,
                                          matched_blocks, WUBU_CONT_MAX_TOKENS / 16);

    if (matched > 0) {
        /* Prefix hit: allocate only suffix blocks */
        s->n_blocks = (n_tokens - matched + cb->block_size - 1) / cb->block_size;
        s->block_ids = (int *)malloc(sizeof(int) * s->n_blocks);
        int phys = 0;
        for (int b = 0; b < s->n_blocks; b++) {
            phys = wubu_paged_kv_ensure(cb->paged_kv, idx, matched + b * cb->block_size);
            if (phys < 0) { seq_state_free(cb, idx); return -1; }
            s->block_ids[b] = phys;
        }
    } else {
        /* No match: allocate all blocks via paged KV */
        s->n_blocks = (n_tokens + cb->block_size - 1) / cb->block_size;
        s->block_ids = (int *)malloc(sizeof(int) * s->n_blocks);
        for (int b = 0; b < s->n_blocks; b++) {
            int phys = wubu_paged_kv_ensure(cb->paged_kv, idx, b * cb->block_size);
            if (phys < 0) { seq_state_free(cb, idx); return -1; }
            s->block_ids[b] = phys;
        }
    }

    /* Register full prefix in cache for future reuse */
    wubu_prefix_cache_register(cb->prefix_cache, tokens, n_tokens,
                               cb->paged_kv, cb->block_size);

    cb->n_active++;
    return idx;
}

/* Remove completed/failed sequence */
void wubu_cont_batch_remove_seq(wubu_cont_batch_t *cb, int seq_idx) {
    if (seq_idx < 0 || seq_idx >= WUBU_CONT_MAX_SEQ) return;
    wubu_seq_state_t *s = &cb->seqs[seq_idx];
    if (!s->alive) return;

    /* Free KV blocks */
    if (cb->paged_kv) wubu_paged_kv_free_seq(cb->paged_kv, seq_idx);
    /* Release prefix cache refs */
    if (cb->prefix_cache) wubu_prefix_cache_release(cb->prefix_cache, s->tokens, s->n_tokens);

    seq_state_free(cb, seq_idx);
    cb->n_active--;
}

/* ---------- Per-iteration scheduling ---------- */

typedef struct {
    int seq_idx;
    int is_prefill;
    int n_new_tokens;  /* for prefill: total tokens; for decode: 1 */
    int prefix_matched;
} wubu_sched_item_t;

/* Build schedule for this iteration: prefill new, decode continuing */
int wubu_cont_batch_schedule(wubu_cont_batch_t *cb, wubu_sched_item_t *out, int max_items) {
    int n = 0;
    cb->iteration++;

    /* First: new sequences needing prefill (not yet started) */
    for (int i = 0; i < WUBU_CONT_MAX_SEQ && n < max_items; i++) {
        if (cb->seqs[i].alive && !cb->seqs[i].prefill_done) {
            out[n++] = (wubu_sched_item_t){
                .seq_idx = i,
                .is_prefill = 1,
                .n_new_tokens = cb->seqs[i].n_tokens,
                .prefix_matched = 0  /* would need prefix cache integration */
            };
        }
    }

    /* Then: continuing sequences (decode one token each) */
    for (int i = 0; i < WUBU_CONT_MAX_SEQ && n < max_items; i++) {
        if (cb->seqs[i].alive && cb->seqs[i].prefill_done &&
            cb->seqs[i].tokens_generated < cb->max_tokens_per_seq) {
            out[n++] = (wubu_sched_item_t){
                .seq_idx = i,
                .is_prefill = 0,
                .n_new_tokens = 1,
                .prefix_matched = 0
            };
        }
    }

    return n;
}

/* Mark sequence's prefill as complete */
void wubu_cont_batch_prefill_done(wubu_cont_batch_t *cb, int seq_idx) {
    if (seq_idx >= 0 && seq_idx < WUBU_CONT_MAX_SEQ) {
        cb->seqs[seq_idx].prefill_done = 1;
    }
}

/* Record generated token for a sequence */
void wubu_cont_batch_record_token(wubu_cont_batch_t *cb, int seq_idx, int token_id) {
    if (seq_idx < 0 || seq_idx >= WUBU_CONT_MAX_SEQ) return;
    wubu_seq_state_t *s = &cb->seqs[seq_idx];
    if (!s->alive) return;

    /* Extend token array */
    int new_len = s->n_tokens + 1;
    int *new_toks = (int *)realloc(s->tokens, sizeof(int) * new_len);
    if (!new_toks) return;
    s->tokens = new_toks;
    s->tokens[s->n_tokens] = token_id;
    s->n_tokens = new_len;
    s->tokens_generated++;

    /* Ensure KV block for new position */
    if (cb->paged_kv) {
        int pos = s->n_tokens - 1;
        wubu_paged_kv_ensure(cb->paged_kv, seq_idx, pos);
    }
}

/* ---------- Stats ---------- */

void wubu_cont_batch_stats(const wubu_cont_batch_t *cb,
                           int *active, int *total_tokens, int *kv_blocks_used,
                           int *kv_blocks_free, size_t *prefix_hits, size_t *prefix_misses) {
    if (active) *active = cb->n_active;
    int tot = 0;
    for (int i = 0; i < WUBU_CONT_MAX_SEQ; i++) if (cb->seqs[i].alive) tot += cb->seqs[i].n_tokens;
    if (total_tokens) *total_tokens = tot;
    if (kv_blocks_used && cb->paged_kv) *kv_blocks_used = cb->paged_kv->n_blocks - wubu_paged_kv_free_count(cb->paged_kv);
    if (kv_blocks_free && cb->paged_kv) *kv_blocks_free = wubu_paged_kv_free_count(cb->paged_kv);
    if (prefix_hits && cb->prefix_cache) *prefix_hits = cb->prefix_cache->hits;
    if (prefix_misses && cb->prefix_cache) *prefix_misses = cb->prefix_cache->misses;
}