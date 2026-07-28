/* Test: wubu_scheduler (Areas H/I — continuous batching + prefix cache). */
#include "wubu_scheduler.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main(void) {
    wubu_sched_t *s = wubu_sched_create(8);
    assert(s != NULL);

    /* Two requests sharing the first 4-token prefix -> both should hit cache. */
    int a[6] = {1, 2, 3, 4, 5, 6};
    int b[5] = {1, 2, 3, 4, 7};
    uint64_t pfx = wubu_prefix_hash(a, 4);   /* shared 4-token prefix */
    wubu_req_t *r1 = wubu_req_create(1, a, 6, 4);
    wubu_req_t *r2 = wubu_req_create(2, b, 5, 4);
    /* Both share a 4-token cacheable prefix -> same hash. */
    r1->prefix_len_cache = 4; r1->prefix_hash = wubu_prefix_hash(a, 4);
    r2->prefix_len_cache = 4; r2->prefix_hash = wubu_prefix_hash(b, 4);
    assert(r1->prefix_hash == pfx && r2->prefix_hash == pfx);  /* same prefix -> same hash */

    wubu_sched_submit(s, r1);
    wubu_sched_submit(s, r2);

    /* Step the scheduler; both should transition PREFILL->DECODE. */
    int active = wubu_sched_step(s);
    printf("after step1: active=%d (expect 2), r1.state=%d r2.state=%d\n",
           active, r1->state, r2->state);
    assert(active == 2);
    assert(r1->state == WUBU_REQ_DECODE && r2->state == WUBU_REQ_DECODE);

    /* Emit tokens; run until both done. */
    int steps = 0;
    while (active > 0 && steps < 20) {
        for (int i = 0; i < s->n; i++)
            if (s->reqs[i]->state == WUBU_REQ_DECODE)
                wubu_req_emit(s->reqs[i], 100 + i);
        active = wubu_sched_step(s);
        steps++;
    }
    printf("finished after %d steps, active=%d\n", steps, active);
    assert(active == 0);
    assert(r1->n_gen == 4 && r2->n_gen == 4);
    printf("ALL SCHEDULER TESTS PASSED\n");
    wubu_sched_free(s);
    return 0;
}
