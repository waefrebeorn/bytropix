/* wubu_slotbank.c -- BRAIN #81 SSD expert slot-bank (LRU paging).
 * See wubu_slotbank.h. */
#include "wubu_slotbank.h"

int wubu_slotbank_init(wubu_slotbank_t *b, int n_slots, int n_experts) {
    if (!b || n_slots <= 0 || n_slots > WUBU_SLOTBANK_MAX_SLOTS) return -1;
    if (n_experts <= 0 || n_experts > WUBU_SLOTBANK_MAX_EXPERTS) return -1;
    b->n_slots = n_slots;
    b->n_experts = n_experts;
    b->clock = 0;
    b->hits = b->misses = b->evictions = b->writebacks = 0;
    for (int i = 0; i < WUBU_SLOTBANK_MAX_SLOTS; i++) {
        b->slots[i].expert_id = -1;
        b->slots[i].last_used = 0;
        b->slots[i].dirty = 0;
    }
    return 0;
}

static int find_expert(const wubu_slotbank_t *b, int expert_id) {
    for (int i = 0; i < b->n_slots; i++)
        if (b->slots[i].expert_id == expert_id) return i;
    return -1;
}

int wubu_slotbank_resident(const wubu_slotbank_t *b, int expert_id) {
    if (!b) return 0;
    return find_expert(b, expert_id) >= 0 ? 1 : 0;
}

int wubu_slotbank_touch(wubu_slotbank_t *b, int expert_id) {
    if (!b || expert_id < 0 || expert_id >= b->n_experts) return -1;
    b->clock++;
    int slot = find_expert(b, expert_id);
    if (slot >= 0) {
        b->slots[slot].last_used = b->clock;   /* LRU refresh */
        b->hits++;
        return slot;
    }
    /* miss: need a slot. Find a free one, else evict LRU. */
    int target = -1;
    for (int i = 0; i < b->n_slots; i++)
        if (b->slots[i].expert_id < 0) { target = i; break; }
    if (target < 0) {
        /* evict least-recently-used */
        uint64_t oldest = UINT64_MAX;
        for (int i = 0; i < b->n_slots; i++) {
            if (b->slots[i].last_used < oldest) { oldest = b->slots[i].last_used; target = i; }
        }
        b->evictions++;
        if (b->slots[target].dirty) b->writebacks++;
    }
    b->slots[target].expert_id = expert_id;
    b->slots[target].last_used = b->clock;
    b->slots[target].dirty = 0;
    b->misses++;
    return target;
}

int wubu_slotbank_mark_dirty(wubu_slotbank_t *b, int expert_id) {
    if (!b) return -1;
    int slot = find_expert(b, expert_id);
    if (slot < 0) return -1;
    b->slots[slot].dirty = 1;
    return 0;
}

uint64_t wubu_slotbank_hits(const wubu_slotbank_t *b) { return b ? b->hits : 0; }
uint64_t wubu_slotbank_misses(const wubu_slotbank_t *b) { return b ? b->misses : 0; }
uint64_t wubu_slotbank_evictions(const wubu_slotbank_t *b) { return b ? b->evictions : 0; }
uint64_t wubu_slotbank_writebacks(const wubu_slotbank_t *b) { return b ? b->writebacks : 0; }
