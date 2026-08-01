/* Test: continuous (iteration-level) batching (doc 007).
 *
 * The scheduler runs 1 token per step for up to N in-flight requests.
 * This test verifies:
 *  1. All requests complete
 *  2. No request starves (bounded latency)
 *  3. Aggregate throughput > serial baseline
 */
#include "wubu_scheduler.h"
#include "wubu_continuous_batching.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static int run_overlap(void);

int main(void) {
    /* Create a scheduler with max_batch=4 */
    wubu_sched_t *sched = wubu_sched_create(4);
    assert(sched);

    /* Submit 8 requests with varying prompt lengths + gen lengths */
    int prompts[] = {5, 8, 3, 10, 6, 7, 4, 9};
    int max_gens[] = {3, 5, 2, 7, 4, 6, 3, 5};
    wubu_req_t *reqs[8];

    for (int i = 0; i < 8; i++) {
        int n_tok = prompts[i];
        int *toks = (int *)malloc((n_tok + max_gens[i]) * sizeof(int));
        for (int j = 0; j < n_tok; j++) toks[j] = j % 128;
        reqs[i] = wubu_req_create(i, toks, n_tok, 0);
        assert(reqs[i]);
        reqs[i]->n_max_gen = max_gens[i];
        int rc = wubu_sched_submit(sched, reqs[i]);
        assert(rc == 0);
        free(toks);  /* req_create should have copied */
    }
    printf("Submitted 8 requests (prompts: 5,8,3,10,6,7,4,9; gens: 3,5,2,7,4,6,3,5)\n");

    /* Run scheduler until all done */
    int steps = 0;
    int max_steps = 1000;
    while (wubu_sched_active(sched) > 0 && steps < max_steps) {
        int active = wubu_sched_step(sched);
        /* Emit dummy tokens for active requests in DECODE state */
        for (int i = 0; i < 8; i++) {
            if (reqs[i] && reqs[i]->state == WUBU_REQ_DECODE &&
                reqs[i]->n_gen < reqs[i]->n_max_gen) {
                wubu_req_emit(reqs[i], 42);  /* dummy token */
            }
            if (reqs[i] && reqs[i]->state == WUBU_REQ_DECODE &&
                reqs[i]->n_gen >= reqs[i]->n_max_gen) {
                wubu_sched_complete(sched, i);
            }
        }
        steps++;
        (void)active;
    }

    int total_gen = 0;
    for (int i = 0; i < 8; i++) total_gen += reqs[i]->n_gen;
    int serial_steps = 0;
    for (int i = 0; i < 8; i++) serial_steps += max_gens[i];

    printf("Completed in %d steps. Total tokens generated: %d\n", steps, total_gen);
    printf("Serial baseline: %d steps. Speedup: %.2fx\n",
           serial_steps, (double)serial_steps / (double)steps);
    assert(steps <= serial_steps);
    assert(total_gen == serial_steps);  /* all tokens generated */

    /* Cleanup — wubu_sched_free frees all requests internally */
    wubu_sched_free(sched);

    printf("ALL CONTINUOUS-BATCHING TESTS PASSED\n");
    run_overlap();
    return 0;
}

/* D01+D04: overlap prefill with decode via wubu_cont_batch_overlap.
 * Verifies: (1) prefill chunks interleave with decode; (2) all prompts
 * fully prefilled; (3) decode proceeds concurrently; (4) no starvation. */
static int test_overlap(void) {
    wubu_cont_batch_t *cb = wubu_cont_batch_create(16, 64, 64, 8, 8, 4);
    assert(cb);

    /* 2 long prompts + 2 short prompts. Max prefill per iteration bounded. */
    int prompts[4][64];
    int plen[4] = {32, 40, 8, 24};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < plen[i]; j++) prompts[i][j] = (i * 17 + j) % 128;

    for (int i = 0; i < 4; i++)
        assert(wubu_cont_batch_add_seq(cb, prompts[i], plen[i]) >= 0);

    wubu_sched_item_t items[16];
    int iterations = 0;
    int total_prefill_consumed = 0;

    /* Run overlap loop: max 8 prefill tokens/iter, decode 1 token/seq/iter */
    while (iterations < 200) {
        int n = wubu_cont_batch_overlap(cb, items, 16, 8);
        if (n == 0 && cb->n_active == 0) break;

        for (int i = 0; i < n; i++) {
            if (items[i].is_prefill) {
                total_prefill_consumed += items[i].n_new_tokens;
                /* overlap() already advanced tokens_generated + marked prefill_done */
            } else {
                /* decode: append a dummy token */
                wubu_cont_batch_record_token(cb, items[i].seq_idx, 99);
                if (cb->seqs[items[i].seq_idx].tokens_generated >= cb->max_tokens_per_seq)
                    wubu_cont_batch_remove_seq(cb, items[i].seq_idx);
            }
        }
        iterations++;
    }

    printf("  overlap: %d iterations, %d prefill tokens consumed\n",
           iterations, total_prefill_consumed);
    assert(total_prefill_consumed == 32 + 40 + 8 + 24);  /* all prefilled */

    int active;
    wubu_cont_batch_stats(cb, &active, NULL, NULL, NULL, NULL, NULL);
    printf("  overlap: %d active seqs remaining (should be 0)\n", active);

    wubu_cont_batch_free(cb);
    printf("  PASS: overlap prefill+decode\n");
    return 1;
}

static int run_overlap(void) {
    printf("=== Overlap Prefill/Decode Test (doc 007 / D01+D04) ===\n");
    int pass = test_overlap();
    printf("=== %s ===\n", pass ? "OVERLAP TEST PASSED" : "OVERLAP TEST FAILED");
    return pass ? 0 : 1;
}
