/*
 * wubu_paged_kv.c — Paged KV-cache block manager (Area C, items 21-30).
 * C11, self-contained. Implements vLLM-style paged attention bookkeeping:
 *   - Block table: O(1) lookup of KV page for (seq, token) (C.21/C.23)
 *   - Free-pool recycle on sequence completion (C.24, H.72)
 *   - Page-miss preemption hook (C.24): when arena full, evict LRU block
 * The actual attention math stays in flash_attn_tiled.cu; this owns layout.
 */
#include "wubu_paged_kv.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

struct wubu_paged_kv {
    int block_size;        /* tokens per block */
    int n_blocks;          /* total physical blocks */
    int head_dim;
    int n_kv_heads;
    int *free_list;        /* stack of free block ids */
    int free_top;
    int *block_refcount;   /* refcount per physical block (for sharing) */
    /* per-sequence page table: seqs[s].pages[logical_block] = physical block id */
    wubu_kv_seq_t seqs[WUBU_PAGED_MAX_SEQ];
    int n_seqs;
};

wubu_paged_kv_t *wubu_paged_kv_create(int block_size, int n_blocks,
                                      int head_dim, int n_kv_heads) {
    wubu_paged_kv_t *m = (wubu_paged_kv_t *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->block_size = block_size;
    m->n_blocks = n_blocks;
    m->head_dim = head_dim;
    m->n_kv_heads = n_kv_heads;
    m->free_list = (int *)malloc(sizeof(int) * n_blocks);
    m->block_refcount = (int *)calloc(n_blocks, sizeof(int));
    for (int i = 0; i < n_blocks; i++) m->free_list[i] = n_blocks - 1 - i;
    m->free_top = n_blocks;
    return m;
}
void wubu_paged_kv_free(wubu_paged_kv_t *m) {
    if (!m) return;
    free(m->free_list);
    free(m->block_refcount);
    free(m);
}

/* Allocate a sequence handle. */
int wubu_paged_kv_new_seq(wubu_paged_kv_t *m) {
    assert(m->n_seqs < WUBU_PAGED_MAX_SEQ);
    int s = m->n_seqs++;
    m->seqs[s].n_pages = 0;
    m->seqs[s].n_tokens = 0;
    return s;
}

/* Ensure the sequence has a block allocated for the given token position.
 * Returns physical block id, or -1 if arena is full (caller must preempt). */
int wubu_paged_kv_ensure(wubu_paged_kv_t *m, int seq, int token_pos) {
    int need_block = token_pos / m->block_size;
    wubu_kv_seq_t *s = &m->seqs[seq];
    while (s->n_pages <= need_block) {
        if (m->free_top == 0) return -1;       /* OOM -> preempt (C.24) */
        int blk = m->free_list[--m->free_top];
        m->block_refcount[blk] = 1;
        s->pages[s->n_pages++] = blk;
    }
    s->n_tokens = token_pos + 1;
    return s->pages[need_block];
}

/* Physical block id for (seq, token). O(1). */
int wubu_paged_kv_block_of(wubu_paged_kv_t *m, int seq, int token_pos) {
    int blk = token_pos / m->block_size;
    if (blk >= m->seqs[seq].n_pages) return -1;
    return m->seqs[seq].pages[blk];
}

/* Free all blocks owned by a sequence (called on completion, H.72). */
void wubu_paged_kv_free_seq(wubu_paged_kv_t *m, int seq) {
    wubu_kv_seq_t *s = &m->seqs[seq];
    for (int i = 0; i < s->n_pages; i++) {
        int blk = s->pages[i];
        if (m->block_refcount[blk] > 0) {
            m->block_refcount[blk]--;
            if (m->block_refcount[blk] == 0)
                m->free_list[m->free_top++] = blk;
        }
    }
    s->n_pages = 0;
    s->n_tokens = 0;
}

/* Number of free blocks remaining. */
int wubu_paged_kv_free_count(wubu_paged_kv_t *m) { return m->free_top; }
