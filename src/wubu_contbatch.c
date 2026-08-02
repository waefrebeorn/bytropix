/*
 * wubu_contbatch.c -- Continuous batching (iterative-level scheduling) (HH04). C11.
 *
 * Convergence (Orca / vLLM continuous batching 7-hop):
 *   - HH04: schedule at iteration (token) granularity, not request granularity.
 *     New requests inject mid-generation; finished requests free slots
 *     immediately. Optimizes throughput (not per-request latency). At home:
 *     the AGI-OS operator runs many CoAgent sweeps concurrently; continuous
 *     batching lets new config-eval requests join the decode batch without
 *     waiting for the current batch → higher effective tok/s under load.
 */
#include "wubu_contbatch.h"
#include <string.h>

int wubu_contbatch_add(wubu_contbatch_t *cb, int req_id, int max_tokens) {
    if (!cb || cb->n_reqs >= WUBU_CONTBATCH_MAX_REQ) return -1;
    wubu_cb_req_t *r = &cb->reqs[cb->n_reqs++];
    r->req_id = req_id;
    r->tokens_generated = 0;
    r->max_tokens = max_tokens;
    r->done = 0;
    return 0;
}

int wubu_contbatch_step(wubu_contbatch_t *cb) {
    if (!cb) return -1;
    int running = 0;
    for (int i = 0; i < cb->n_reqs; i++) {
        wubu_cb_req_t *r = &cb->reqs[i];
        if (r->done) continue;
        r->tokens_generated++;
        if (r->tokens_generated >= r->max_tokens) r->done = 1;
        else running++;
    }
    cb->step++;
    cb->running = running;
    return running;
}

float wubu_contbatch_tput(const wubu_contbatch_t *cb) {
    if (!cb || cb->step == 0) return 0.0f;
    int total = 0;
    for (int i = 0; i < cb->n_reqs; i++) total += cb->reqs[i].tokens_generated;
    return (float)total / (float)cb->step;
}
