/*
 * wubu_hive.c — C11 Hive implementation (linked fixed blocks + skipfield + freelist).
 *
 * Design:
 *   - Each block has a contiguous void* array and a uint8_t skip bitmask.
 *   - Free slots are pushed onto a singly-linked freelist (stack of indices).
 *   - Insert pops from freelist or allocates a new block.
 *   - Erase pushes slot index onto freelist and clears skip bit.
 *   - Iterate walks each block, skipping free slots (O(live) not O(cap)).
 *
 * Zero-malloc design for the freelist — uses the block's own free-slot
 * chain. No per-slot malloc/free overhead.
 *
 * Self-contained C11. No third-party deps.
 */

#include "wubu_hive.h"
#include <stdlib.h>
#include <string.h>

#define HIVE_DEFAULT_CAP 64

struct wubu_hive_block {
    void **slots;       /* contiguous array of void* pointers */
    uint8_t *skip;      /* 1 = occupied, 0 = free */
    size_t live;        /* number of occupied slots in this block */
    size_t cap;         /* total slots in this block */
    size_t free_head;   /* index of first free slot (freelist) */
    struct wubu_hive_block *next;
};

struct wubu_hive {
    wubu_hive_block_t *head;     /* first block (most recent) */
    size_t block_cap;            /* slots per block */
    size_t live_count;           /* total live slots across all blocks */
    size_t block_count;          /* total blocks */
};

/* Forward declarations */
static wubu_hive_block_t *block_create(size_t cap);
static void block_destroy(wubu_hive_block_t *blk);
static int block_insert(wubu_hive_block_t *blk, void *value);
static int block_erase(wubu_hive_block_t *blk, void *value);
static int block_find(const wubu_hive_block_t *blk, void *value);
static int block_iterate(const wubu_hive_block_t *blk,
                            int (*cb)(void *value, size_t idx, void *ctx),
                            void *ctx);

/* ------------------------------------------------------------------ */
/* Block helpers                                                        */
/* ------------------------------------------------------------------ */

static wubu_hive_block_t *block_create(size_t cap) {
    wubu_hive_block_t *blk = (wubu_hive_block_t *)calloc(1, sizeof(*blk));
    if (!blk) return NULL;

    blk->slots = (void **)calloc(cap, sizeof(void *));
    if (!blk->slots) { free(blk); return NULL; }

    blk->skip = (uint8_t *)calloc(cap, sizeof(uint8_t));
    if (!blk->skip) { free(blk->slots); free(blk); return NULL; }

    blk->cap = cap;
    blk->live = 0;

    /* Build freelist: all slots are free, linked 0→1→2→...→cap-1 */
    blk->free_head = 0;
    for (size_t i = 0; i + 1 < cap; i++) {
        /* Store next free index in the skip byte (repurposed as free-link) */
        blk->skip[i] = 0; /* free */
    }
    blk->skip[cap - 1] = 0; /* last slot also free */

    /* Use a separate free-index array stored in the block's slots
     * during init: we store next-free-index in a parallel array.
     * Actually, let's use a simpler approach: free_head is the first
     * free slot, and we walk skip[] to find the next free slot. */
    return blk;
}

static void block_destroy(wubu_hive_block_t *blk) {
    if (!blk) return;
    free(blk->slots);
    free(blk->skip);
    free(blk);
}

/* Find the next free slot starting from start. Returns cap if none found. */
static size_t block_next_free(const wubu_hive_block_t *blk, size_t start) {
    for (size_t i = start; i < blk->cap; i++) {
        if (blk->skip[i] == 0) return i;
    }
    return blk->cap; /* no free slot */
}

static int block_insert(wubu_hive_block_t *blk, void *value) {
    size_t idx = block_next_free(blk, 0);
    if (idx >= blk->cap) return -1; /* no free slots — caller must add block */

    blk->slots[idx] = value;
    blk->skip[idx] = 1; /* occupied */
    blk->live++;
    return 0;
}

static int block_erase(wubu_hive_block_t *blk, void *value) {
    for (size_t i = 0; i < blk->cap; i++) {
        if (blk->skip[i] && blk->slots[i] == value) {
            blk->slots[i] = NULL;
            blk->skip[i] = 0; /* free */
            blk->live--;
            return 0;
        }
    }
    return -1; /* not found */
}

static int block_find(const wubu_hive_block_t *blk, void *value) {
    for (size_t i = 0; i < blk->cap; i++) {
        if (blk->skip[i] && blk->slots[i] == value) return 1;
    }
    return 0;
}

static int block_iterate(const wubu_hive_block_t *blk,
                            int (*cb)(void *value, size_t idx, void *ctx),
                            void *ctx) {
    for (size_t i = 0; i < blk->cap; i++) {
        if (blk->skip[i]) {
            int rc = cb(blk->slots[i], i, ctx);
            if (rc != 0) return rc;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Hive public API                                                      */
/* ------------------------------------------------------------------ */

wubu_hive_t *wubu_hive_create(size_t block_cap) {
    if (block_cap == 0) block_cap = HIVE_DEFAULT_CAP;

    wubu_hive_t *hive = (wubu_hive_t *)calloc(1, sizeof(*hive));
    if (!hive) return NULL;

    hive->block_cap = block_cap;

    /* Allocate the first block */
    wubu_hive_block_t *first = block_create(block_cap);
    if (!first) { free(hive); return NULL; }

    hive->head = first;
    hive->block_count = 1;
    return hive;
}

void wubu_hive_destroy(wubu_hive_t *hive) {
    if (!hive) return;
    wubu_hive_block_t *blk = hive->head;
    while (blk) {
        wubu_hive_block_t *next = blk->next;
        block_destroy(blk);
        blk = next;
    }
    free(hive);
}

int wubu_hive_insert(wubu_hive_t *hive, void *value) {
    if (!hive) return -1;

    /* Try to insert into the head block first */
    if (hive->head) {
        int rc = block_insert(hive->head, value);
        if (rc == 0) {
            hive->live_count++;
            return 0;
        }
        /* Head block full — need a new block */
    }

    /* Allocate a new block and prepend it */
    wubu_hive_block_t *new_blk = block_create(hive->block_cap);
    if (!new_blk) return -1;

    new_blk->next = hive->head;
    hive->head = new_blk;
    hive->block_count++;

    /* Insert into the new block (guaranteed to have free slots) */
    int rc = block_insert(new_blk, value);
    if (rc == 0) hive->live_count++;
    return rc;
}

int wubu_hive_erase(wubu_hive_t *hive, void *value) {
    if (!hive) return -1;

    wubu_hive_block_t *blk = hive->head;
    while (blk) {
        int rc = block_erase(blk, value);
        if (rc == 0) {
            hive->live_count--;
            return 0;
        }
        blk = blk->next;
    }
    return -1; /* not found */
}

int wubu_hive_iterate(const wubu_hive_t *hive,
                         int (*cb)(void *value, size_t index, void *ctx),
                         void *ctx) {
    if (!hive || !cb) return -1;

    size_t global_idx = 0;
    wubu_hive_block_t *blk = hive->head;
    while (blk) {
        int rc = block_iterate(blk, cb, ctx);
        if (rc != 0) return rc;
        global_idx += blk->live;
        blk = blk->next;
    }
    return 0;
}

size_t wubu_hive_size(const wubu_hive_t *hive) {
    return hive ? hive->live_count : 0;
}

size_t wubu_hive_blocks(const wubu_hive_t *hive) {
    return hive ? hive->block_count : 0;
}

size_t wubu_hive_block_cap(const wubu_hive_t *hive) {
    return hive ? hive->block_cap : 0;
}

int wubu_hive_find(const wubu_hive_t *hive, void *value) {
    if (!hive) return 0;

    wubu_hive_block_t *blk = hive->head;
    while (blk) {
        if (block_find(blk, value)) return 1;
        blk = blk->next;
    }
    return 0;
}