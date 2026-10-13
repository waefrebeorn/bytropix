#ifndef WUBU_SLOTBANK_H
#define WUBU_SLOTBANK_H
#include <stdint.h>

/* wubu_slotbank.h -- BRAIN #81 SSD expert slot-bank pages 256 experts on
 * small RAM.
 *
 * A slot-bank holds N_SLOTS expert weights in RAM (a small fixed footprint)
 * while the full expert set (e.g. 256 experts) lives on SSD. On access, an
 * expert is paged into a free slot; if all slots are full, the least-recently-
 * used expert is evicted (written back / dropped) to make room. This lets a
 * model with far more experts than fit in RAM run by paging on demand.
 * Tracks hit/miss/eviction stats. Deterministic LRU, no allocation beyond
 * the fixed slot array, real work (no stubs). */

#define WUBU_SLOTBANK_MAX_EXPERTS 1024
#define WUBU_SLOTBANK_MAX_SLOTS   64

typedef struct {
    int      expert_id;    /* which expert occupies this slot, -1 = empty */
    uint64_t last_used;    /* LRU timestamp (access counter) */
    int      dirty;        /* 1 = modified since load (needs writeback) */
} wubu_slot_t;

typedef struct {
    wubu_slot_t slots[WUBU_SLOTBANK_MAX_SLOTS];
    int      n_slots;
    int      n_experts;    /* total experts (on SSD) */
    uint64_t clock;        /* monotonic access counter for LRU */
    /* stats */
    uint64_t hits;         /* expert already in a slot */
    uint64_t misses;       /* expert had to be paged in */
    uint64_t evictions;    /* LRU evictions */
    uint64_t writebacks;   /* dirty evictions */
} wubu_slotbank_t;

/* Initialise a bank with `n_slots` RAM slots and `n_experts` total experts. */
int wubu_slotbank_init(wubu_slotbank_t *b, int n_slots, int n_experts);

/* Access an expert: returns the slot index holding it (paging in + evicting
 * LRU if needed), or -1 on bad input. Counts a hit (already resident) or
 * miss (paged in). */
int wubu_slotbank_touch(wubu_slotbank_t *b, int expert_id);

/* Is the expert currently resident in a slot? 1/0. */
int wubu_slotbank_resident(const wubu_slotbank_t *b, int expert_id);

/* Mark the expert dirty (modified, needs writeback on eviction). */
int wubu_slotbank_mark_dirty(wubu_slotbank_t *b, int expert_id);

/* Stats accessors. */
uint64_t wubu_slotbank_hits(const wubu_slotbank_t *b);
uint64_t wubu_slotbank_misses(const wubu_slotbank_t *b);
uint64_t wubu_slotbank_evictions(const wubu_slotbank_t *b);
uint64_t wubu_slotbank_writebacks(const wubu_slotbank_t *b);

#endif
