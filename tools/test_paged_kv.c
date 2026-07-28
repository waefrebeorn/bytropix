/* Test: wubu_paged_kv (Area C — paged attention block manager). */
#include "wubu_paged_kv.h"
#include <stdio.h>
#include <assert.h>

int main(void) {
    /* 4 tokens/block, 8 blocks, head_dim 16, 2 kv heads. */
    wubu_paged_kv_t *m = wubu_paged_kv_create(4, 8, 16, 2);
    assert(m != NULL);
    assert(wubu_paged_kv_free_count(m) == 8);

    int s = wubu_paged_kv_new_seq(m);
    /* Write 10 tokens -> needs 3 blocks (0-3, 4-7, 8-9). */
    for (int t = 0; t < 10; t++) {
        int blk = wubu_paged_kv_ensure(m, s, t);
        assert(blk >= 0);   /* arena has room */
    }
    assert(wubu_paged_kv_block_of(m, s, 0)  == wubu_paged_kv_block_of(m, s, 3));
    assert(wubu_paged_kv_block_of(m, s, 4)  == wubu_paged_kv_block_of(m, s, 7));
    assert(wubu_paged_kv_block_of(m, s, 8)  != wubu_paged_kv_block_of(m, s, 3));
    printf("seq blocks allocated, free now=%d (expect 5)\n", wubu_paged_kv_free_count(m));
    assert(wubu_paged_kv_free_count(m) == 5);

    /* Free the seq -> blocks return to pool. */
    wubu_paged_kv_free_seq(m, s);
    printf("after free_seq, free=%d (expect 8)\n", wubu_paged_kv_free_count(m));
    assert(wubu_paged_kv_free_count(m) == 8);

    /* OOM path: exhaust arena, ensure returns -1. */
    int s2 = wubu_paged_kv_new_seq(m);
    int exhausted = 0;
    for (int t = 0; t < 100; t++) {
        if (wubu_paged_kv_ensure(m, s2, t) < 0) { exhausted = 1; break; }
    }
    printf("arena exhaustion detected=%d (expect 1)\n", exhausted);
    assert(exhausted);

    wubu_paged_kv_free(m);
    printf("ALL PAGED-KV TESTS PASSED\n");
    return 0;
}
