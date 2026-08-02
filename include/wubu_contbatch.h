/*
 * wubu_contbatch.h -- Continuous batching (iterative-level scheduling) (HH04).
 */
#ifndef WUBU_CONTBATCH_H
#define WUBU_CONTBATCH_H

#define WUBU_CONTBATCH_MAX_REQ 128

typedef struct {
    int req_id;
    int tokens_generated;
    int max_tokens;
    int done;
} wubu_cb_req_t;

typedef struct {
    wubu_cb_req_t reqs[WUBU_CONTBATCH_MAX_REQ];
    int n_reqs;
    int step;            /* current decode iteration */
    int running;         /* number of in-flight requests this step */
} wubu_contbatch_t;

/* Register a new request (can join mid-generation). */
int  wubu_contbatch_add(wubu_contbatch_t *cb, int req_id, int max_tokens);
/* Advance one decode step: all non-done requests produce 1 token; done ones
   free their slot immediately (no waiting for batch completion). */
int  wubu_contbatch_step(wubu_contbatch_t *cb);
/* Effective throughput proxy: total tokens produced / steps elapsed. */
float wubu_contbatch_tput(const wubu_contbatch_t *cb);

#endif