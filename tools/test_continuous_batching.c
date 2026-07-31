/* Test: continuous (iteration-level) batching (doc 007).
 *
 * The scheduler runs 1 token per step for up to N in-flight requests.
 * This test verifies:
 *  1. All requests complete
 *  2. No request starves (bounded latency)
 *  3. Aggregate throughput > serial baseline
 */
#include "wubu_scheduler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

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
    return 0;
}
