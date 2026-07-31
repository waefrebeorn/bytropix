/* Test: chunked prefill + disaggregated PD (doc D03/D04). */
#include "wubu_chunked_prefill.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

int main(void) {
    /* Create with chunk size=128 */
    wubu_chunked_prefill_t *c = wubu_chunked_prefill_create(128);
    assert(c);

    /* Test 1: submit a 512-token prompt → 4 chunks of 128 */
    int job = wubu_chunked_prefill_submit(c, 512);
    assert(job >= 0);
    printf("Submitted 512-token job (id=%d)\n", job);

    int total_chunks = 0;
    while (!wubu_chunked_prefill_is_done(c, job)) {
        int chunk = wubu_chunked_prefill_next_chunk(c, job);
        if (chunk <= 0) break;
        total_chunks++;
        printf("  Chunk %d: %d tokens\n", total_chunks, chunk);
    }
    assert(total_chunks == 4);
    assert(wubu_chunked_prefill_is_done(c, job));
    printf("512 tokens → %d chunks of 128\n", total_chunks);

    /* Test 2: progress tracking */
    int job2 = wubu_chunked_prefill_submit(c, 300);
    assert(job2 >= 0);
    assert(wubu_chunked_prefill_progress(c, job2) == 0.0f);
    wubu_chunked_prefill_next_chunk(c, job2);  /* 128 */
    assert(wubu_chunked_prefill_progress(c, job2) > 0.0f && wubu_chunked_prefill_progress(c, job2) < 1.0f);
    wubu_chunked_prefill_next_chunk(c, job2);  /* 128 */
    wubu_chunked_prefill_next_chunk(c, job2);  /* 44 remaining */
    assert(wubu_chunked_prefill_progress(c, job2) == 1.0f);
    printf("Progress tracking: 0%% → partial → 100%%\n");

    /* Test 3: non-multiple chunks (301 tokens → 3 chunks: 128+128+45) */
    int job3 = wubu_chunked_prefill_submit(c, 301);
    int chunks3 = 0;
    while (!wubu_chunked_prefill_is_done(c, job3)) {
        int chunk = wubu_chunked_prefill_next_chunk(c, job3);
        if (chunk <= 0) break;
        chunks3++;
    }
    assert(chunks3 == 3);  /* ceil(301/128) = 3 */
    printf("301 tokens → %d chunks (128+128+45)\n", chunks3);

    /* Test 4: schedule with decode budget */
    wubu_chunked_prefill_t *c2 = wubu_chunked_prefill_create(64);
    int j1 = wubu_chunked_prefill_submit(c2, 200);  /* 4 chunks: 64*3+8 */
    int j2 = wubu_chunked_prefill_submit(c2, 100);   /* 2 chunks: 64+36 */
    int out_chunks[64]; int out_decode;
    int active = wubu_chunked_prefill_schedule(c2, 16, out_chunks, &out_decode);
    printf("Schedule: active=%d decode_tokens=%d\n", active, out_decode);
    assert(active > 0);

    wubu_chunked_prefill_free(c);
    wubu_chunked_prefill_free(c2);
    printf("ALL CHUNKED-PREFILL TESTS PASSED\n");
    return 0;
}
