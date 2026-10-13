/* test_slotbank.c -- BRAIN #81 SSD expert slot-bank pages 256 experts on
 * small RAM. Proves: pages 256 experts through a tiny slot set, LRU eviction,
 * hit/miss/eviction/writeback stats, dirty writeback on eviction, fail-closed. */
#include "wubu_slotbank.h"
#include <stdio.h>

static int fails = 0, checks = 0;
#define CHECK(cond, msg) do { checks++; if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } else { printf("  [PASS] %s\n", msg); } } while (0)

int main(void) {
    /* 4 RAM slots, 256 experts on SSD */
    wubu_slotbank_t b;
    CHECK(wubu_slotbank_init(&b, 4, 256) == 0, "init 4 slots / 256 experts ok");
    CHECK(wubu_slotbank_init(&b, 0, 256) == -1, "zero slots rejected");
    CHECK(wubu_slotbank_init(&b, 4, 0) == -1, "zero experts rejected");
    CHECK(wubu_slotbank_init(&b, 9999, 256) == -1, "oversize slots rejected");

    /* access 4 distinct experts: all fit, all misses, no eviction */
    for (int e = 0; e < 4; e++) wubu_slotbank_touch(&b, e);
    CHECK(wubu_slotbank_misses(&b) == 4, "4 distinct -> 4 misses");
    CHECK(wubu_slotbank_evictions(&b) == 0, "no eviction yet (all fit)");
    CHECK(wubu_slotbank_resident(&b, 0) == 1, "expert 0 resident");

    /* re-access expert 0: HIT (still resident) */
    wubu_slotbank_touch(&b, 0);
    CHECK(wubu_slotbank_hits(&b) == 1, "re-access -> hit");

    /* access a 5th expert: evicts LRU (expert 1, least recently used) */
    wubu_slotbank_touch(&b, 100);
    CHECK(wubu_slotbank_evictions(&b) == 1, "5th distinct -> 1 eviction");
    CHECK(wubu_slotbank_resident(&b, 1) == 0, "LRU expert 1 evicted");
    CHECK(wubu_slotbank_resident(&b, 100) == 1, "new expert 100 resident");
    /* expert 0 was just touched (recent), should still be resident */
    CHECK(wubu_slotbank_resident(&b, 0) == 1, "recently-used expert 0 survives");

    /* dirty writeback: mark expert 100 dirty, then evict it -> writeback */
    wubu_slotbank_mark_dirty(&b, 100);
    /* touch 3 new experts to force eviction of the LRU (which will be 100
     * if it's now the oldest). First refresh 0,2,3 by touching them. */
    wubu_slotbank_touch(&b, 0); wubu_slotbank_touch(&b, 2); wubu_slotbank_touch(&b, 3);
    /* now 100 is the LRU; touching a new expert evicts it (dirty -> writeback) */
    wubu_slotbank_touch(&b, 200);
    CHECK(wubu_slotbank_writebacks(&b) >= 1, "dirty eviction triggers writeback");

    /* page through all 256 experts (stress): every access valid */
    wubu_slotbank_t big;
    wubu_slotbank_init(&big, 8, 256);
    int ok = 1;
    for (int e = 0; e < 256; e++) {
        int slot = wubu_slotbank_touch(&big, e);
        if (slot < 0 || slot >= 8) { ok = 0; break; }
    }
    CHECK(ok == 1, "paged all 256 experts through 8 slots, all valid");
    CHECK(wubu_slotbank_misses(&big) == 256, "256 distinct -> 256 misses");
    CHECK(wubu_slotbank_evictions(&big) == 248, "248 evictions (256-8)");

    /* fail-closed */
    CHECK(wubu_slotbank_touch(&b, -1) == -1, "negative expert -> -1");
    CHECK(wubu_slotbank_touch(&b, 256) == -1, "out-of-range expert -> -1");
    CHECK(wubu_slotbank_touch(NULL, 0) == -1, "NULL bank -> -1");
    CHECK(wubu_slotbank_resident(NULL, 0) == 0, "NULL resident -> 0");
    CHECK(wubu_slotbank_mark_dirty(&b, 999) == -1, "mark dirty non-resident -> -1");

    printf("\n=== test_slotbank: %d checks, %d failures ===\n", checks, fails);
    return fails ? 1 : 0;
}
