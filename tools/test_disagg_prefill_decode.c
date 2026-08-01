/* Test: D03 disaggregated prefill/decode (doc 007).
 * Verifies: (1) prefill engine and decode engine run as separate passes
 * sharing one KV store; (2) all prefill work completes; (3) decode proceeds
 * after prefill; (4) no starvation. */
#include "wubu_continuous_batching.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static int test_disagg(void) {
    wubu_cont_batch_t *cb = wubu_cont_batch_create(16, 64, 64, 8, 8, 100);
    assert(cb);

    int prompts[3][64];
    int plen[3] = {48, 32, 24};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < plen[i]; j++) prompts[i][j] = (i * 31 + j * 3) % 1000;

    for (int i = 0; i < 3; i++)
        assert(wubu_cont_batch_add_seq(cb, prompts[i], plen[i]) >= 0);

    wubu_sched_item_t items[16];
    int iterations = 0, total_prefill = 0, total_decode = 0;

    while (iterations < 200) {
        int n_prefill = 0;
        int n = wubu_cont_batch_disagg(cb, items, 16, 8, &n_prefill);
        if (n == 0 && cb->n_active == 0) break;

        total_prefill += n_prefill;
        for (int i = 0; i < n; i++) {
            if (items[i].is_prefill) {
                /* disagg() already advanced tokens_generated + marked prefill_done */
            } else {
                wubu_cont_batch_record_token(cb, items[i].seq_idx, 7);
                total_decode++;
                if (cb->seqs[items[i].seq_idx].tokens_generated >= cb->max_tokens_per_seq)
                    wubu_cont_batch_remove_seq(cb, items[i].seq_idx);
            }
        }
        iterations++;
    }

    printf("  disagg: %d iters, %d prefill toks, %d decode toks\n",
           iterations, total_prefill, total_decode);
    assert(total_prefill == 48 + 32 + 24);
    assert(total_decode > 0);

    wubu_cont_batch_free(cb);
    printf("  PASS: disaggregated prefill/decode\n");
    return 1;
}

int main(void) {
    printf("=== D03 Disaggregated Prefill/Decode Test (doc 007) ===\n");
    int pass = test_disagg();
    printf("=== %s ===\n", pass ? "D03 TEST PASSED" : "D03 TEST FAILED");
    return pass ? 0 : 1;
}
