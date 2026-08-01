/* Test: A07b priority-based KV eviction (importance + LRU hybrid).
 * Verifies: (1) important-but-stale block is RETAINED vs fresh-but-trivial;
 * (2) lowest-score block is selected first for eviction; (3) recency decay
 * works. */
#include "wubu_kv_evict.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main(void) {
    printf("=== A07b KV Eviction Test ===\n");

    wubu_kv_evict_t *e = wubu_kv_evict_create(0.9f);
    assert(e);

    /* Three blocks: A = recent + important, B = stale + important, C = recent + trivial */
    wubu_kv_evict_track(e, 1, 10.0f);   /* A: important */
    wubu_kv_evict_track(e, 2, 10.0f);   /* B: important */
    wubu_kv_evict_track(e, 3, 0.1f);    /* C: trivial */

    wubu_kv_evict_touch(e, 1);
    wubu_kv_evict_touch(e, 3);
    /* B is never touched again -> becomes stale */

    /* Decay several steps so B's recency EMA drops */
    for (int i = 0; i < 10; i++) {
        wubu_kv_evict_touch(e, 1);
        wubu_kv_evict_touch(e, 3);
        wubu_kv_evict_tick(e);
    }

    int victims[8];
    int nv = wubu_kv_evict_select(e, victims, 3);
    printf("  selected %d victims; first to evict = block %d\n", nv, victims[0]);

    /* A = recent+important, B = stale+important, C = recent+trivial.
     * Eviction order (highest score first): C (trivial) then B (stale imp) then
     * A (recent imp). The importance-respecting policy must evict trivial C
     * before important A/B, and among important ones, the stale B before fresh A. */
    assert(nv == 3);
    assert(victims[0] == 3);  /* C: trivial => evicted first despite being recent */
    assert(victims[2] == 1);  /* A: recent+important => retained longest */
    /* B (stale but important) evicted before A (fresh+important) */
    int b_pos = -1, a_pos = -1;
    for (int i = 0; i < nv; i++) { if (victims[i] == 2) b_pos = i; if (victims[i] == 1) a_pos = i; }
    assert(b_pos >= 0 && a_pos >= 0 && b_pos < a_pos);

    /* Dropping works */
    wubu_kv_evict_drop(e, 1);
    assert(wubu_kv_evict_count(e) == 2);

    wubu_kv_evict_free(e);
    printf("ALL A07b KV-EVICTION TESTS PASSED\n");
    return 0;
}
