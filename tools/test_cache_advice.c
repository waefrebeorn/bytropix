/* Test: wubu_cache_advice (Round-2 #111 — ML-advice eviction). */
#include "wubu_cache_advice.h"
#include <stdio.h>
#include <assert.h>

int main(void) {
    wubu_cache_advice_t *a = wubu_cache_advice_create(4);
    assert(a != NULL);
    /* Hot block 1 touched many times; cold block 99 touched once. */
    for (int t = 0; t < 20; t++) wubu_cache_advice_touch(a, 1, t);
    wubu_cache_advice_touch(a, 2, 21);
    wubu_cache_advice_touch(a, 3, 22);
    wubu_cache_advice_touch(a, 4, 23);
    /* cache full (cap 4). Next miss should evict the LOWEST-value block.
     * Block 2,3,4 each freq=1; block 1 freq=20. So evict one of 2/3/4. */
    int ev = wubu_cache_advice_touch(a, 99, 24);
    printf("evicted block=%d (expect one of 2,3,4; NOT 1)\n", ev);
    assert(ev != 1 && ev >= 2 && ev <= 4);
    /* Block 1 must still be resident (high value). */
    int hit1 = wubu_cache_advice_has(a, 1);
    assert(hit1);
    printf("hot block 1 retained under pressure: OK (count=%d)\n", wubu_cache_advice_count(a));

    wubu_cache_advice_free(a);

    /* DA edge case: cap==0 must return NULL, not crash on malloc(0). */
    wubu_cache_advice_t *z = wubu_cache_advice_create(0);
    printf("create(0) -> %s (expect NULL)\n", z ? "non-NULL" : "NULL");
    assert(z == NULL);

    printf("ALL CACHE-ADVICE TESTS PASSED\n");
    return 0;
}
