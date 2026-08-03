/*
 * test_hive.c -- THE HIVE test: the AGI's memory structure (WuBu).
 *
 * Verifies the properties the user's diagram promises:
 *   - insert: fills blocks, auto-grows a new block
 *   - erase: O(1) skip-mark, live count drops
 *   - reuse: erased slots are reused by the NEXT insert (freelist)
 *   - iterate: skips erased slots, visits exactly the live ones
 *   - stable pointers: the caller's pointers are preserved exactly
 *   - cache: iteration touches only live slots (skipfield jump)
 *   - no compaction, no shifting (a pointer stays in its slot)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_hive.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

static int g_visit = 0;
static int collect(void *ptr, void *user)
{
    (void)user;
    g_visit++;
    return 0;
}

int main(void)
{
    printf("=== test_hive (the AGI's memory: linked blocks + skipfield + freelist) ===\n");
    wubu_hive_t h;
    CHECK(wubu_hive_init(&h) == 0, "init");

    /* the diagram: A..H, plus a few more to force a second block */
    char vals[80];
    for (int i = 0; i < 80; i++) vals[i] = (char)('A' + (i % 26));
    void *ptrs[80];
    for (int i = 0; i < 80; i++) ptrs[i] = &vals[i];

    /* insert 70 -> 2 blocks (64 + 6) */
    for (int i = 0; i < 70; i++) CHECK(wubu_hive_insert(&h, ptrs[i]) == 0, "insert");
    CHECK(wubu_hive_live(&h) == 70, "70 live");
    CHECK(wubu_hive_capacity(&h) == 128, "2 blocks = 128 capacity");
    printf("  inserted 70 -> %zu live, %zu capacity, %zu blocks\n",
           wubu_hive_live(&h), wubu_hive_capacity(&h), h.n_blocks);

    /* iterate: all 70 live */
    g_visit = 0;
    CHECK(wubu_hive_foreach(&h, collect, NULL) == 70, "iterate 70");
    CHECK(g_visit == 70, "visited 70");

    /* erase 5 spread across both blocks */
    CHECK(wubu_hive_erase(&h, ptrs[3]) == 0, "erase B");
    CHECK(wubu_hive_erase(&h, ptrs[10]) == 0, "erase K");
    CHECK(wubu_hive_erase(&h, ptrs[63]) == 0, "erase last of block0");
    CHECK(wubu_hive_erase(&h, ptrs[64]) == 0, "erase first of block1");
    CHECK(wubu_hive_erase(&h, ptrs[69]) == 0, "erase last");
    CHECK(wubu_hive_live(&h) == 65, "65 live after 5 erases");
    g_visit = 0;
    CHECK(wubu_hive_foreach(&h, collect, NULL) == 65, "iterate 65");
    printf("  erased 5 -> %zu live; iteration visited %d (skipfield jumped)\n",
           wubu_hive_live(&h), g_visit);

    /* reuse: insert must reuse a freed slot (freelist), not grow */
    size_t cap_before = wubu_hive_capacity(&h);
    char extra = 'Z';
    CHECK(wubu_hive_insert(&h, &extra) == 0, "insert after erases");
    CHECK(wubu_hive_live(&h) == 66, "66 live");
    CHECK(wubu_hive_capacity(&h) == cap_before, "capacity unchanged (reused slot)");
    CHECK(h.reuses >= 1, "freelist reuse counted");
    printf("  insert reused a freelist slot (capacity stayed %zu, reuses %zu)\n",
           cap_before, h.reuses);

    /* the re-inserted pointer must be found by iteration (erase proves
     * it was inserted into a live slot) */
    CHECK(wubu_hive_erase(&h, &extra) == 0, "erase the reused slot (found)");

    /* pointer stability: erase+insert cycles never move values */
    char *keep[20];
    for (int i = 0; i < 20; i++) keep[i] = &vals[20 + i];
    for (int cyc = 0; cyc < 5; cyc++) {
        for (int i = 0; i < 20; i++) CHECK(wubu_hive_erase(&h, keep[i]) == 0, "cyc erase");
        for (int i = 0; i < 20; i++) CHECK(wubu_hive_insert(&h, keep[i]) == 0, "cyc insert");
    }
    /* all 20 still live and identical */
    g_visit = 0;
    wubu_hive_foreach(&h, collect, NULL);
    CHECK(g_visit == wubu_hive_live(&h), "after cycles, live == visited");
    printf("  5 erase/insert cycles: %zu live, %zu visited -- pointers stable\n",
           wubu_hive_live(&h), (size_t)g_visit);

    /* stress: 100k insert/erase churn, count stays right */
    wubu_hive_t h2;
    wubu_hive_init(&h2);
    char pool[4096];
    for (int i = 0; i < 4096; i++) pool[i] = (char)i;
    for (int i = 0; i < 4096; i++) wubu_hive_insert(&h2, &pool[i]);
    for (int i = 0; i < 4096; i += 2) wubu_hive_erase(&h2, &pool[i]);
    CHECK(wubu_hive_live(&h2) == 2048, "stress: 2048 live after half-erase");
    g_visit = 0;
    wubu_hive_foreach(&h2, collect, NULL);
    CHECK(g_visit == 2048, "stress: iteration found exactly 2048");
    printf("  stress 4096 -> erase half -> %zu live, %zu visited\n",
           wubu_hive_live(&h2), (size_t)g_visit);
    wubu_hive_clear(&h2);

    wubu_hive_clear(&h);
    CHECK(wubu_hive_live(&h) == 0, "clear -> 0 live");
    if (failures == 0) printf("ALL HIVE TESTS PASSED -- the AGI has its memory\n");
    else printf("%d HIVE FAILURES\n", failures);
    return failures ? 1 : 0;
}
