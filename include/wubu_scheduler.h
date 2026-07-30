#ifndef WUBU_SCHEDULER_H
#define WUBU_SCHEDULER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Continuous (iteration-level) scheduler for multi-request batching.
 * Research: doc 007 (Anyscale continuous batching); doc 010 (prefix KV reuse).
 * Model-agnostic: operates on int token ids only, never touches model weights. */

#define SCHED_MAX_REQ 1024
#define SCHED_MAX_BATCH 16

typedef enum {
    WUBU_REQ_PREFILL = 0,
    WUBU_REQ_DECODE  = 1,
    WUBU_REQ_DONE    = 2
} wubu_req_state_t;

typedef struct wubu_req {
    int id;
    int prefix_len;       /* tokens in the shared prefix */
    int n_prompt;         /* total prompt tokens (prefix + unique) */
    int n_gen;            /* tokens generated so far */
    int n_max_gen;        /* max tokens to generate */
    int *tokens;          /* full token sequence [n_prompt + n_gen] */
    int seq_len;          /* current length = n_prompt + n_gen */
    wubu_req_state_t state;
    /* Prefix cache integration (doc 010) */
    int  prefix_len_cache;  /* how many leading tokens hit the prefix cache */
    uint64_t prefix_hash;   /* hash of the shared prefix */
    /* Scheduling bookkeeping */
    int  last_decode_pos;   /* position of last token decoded this step */
} wubu_req_t;

typedef struct wubu_sched {
    wubu_req_t *reqs[SCHED_MAX_REQ];
    int n;              /* number of submitted requests */
    int max_batch;
    int policy;         /* 0=FIFO, 1=ROUND_ROBIN */
    int rr_index;
} wubu_sched_t;

/* Create a scheduler with a given max concurrent batch size. */
wubu_sched_t *wubu_sched_create(int max_batch);
void wubu_sched_free(wubu_sched_t *s);

/* Submit a request. tokens[0..n_prompt-1] = prompt; will generate up to
 * n_max_gen tokens. Returns 0 on success, -1 on capacity. */
int wubu_sched_submit(wubu_sched_t *s, wubu_req_t *req);

/* Step the scheduler: advances all active requests by one iteration.
 * Transitions PREFILL->DECODE when the prompt is fully consumed.
 * Returns number of active requests still in flight. */
int wubu_sched_step(wubu_sched_t *s);

/* Mark a request as complete (e.g. after hitting EOS or max tokens). */
void wubu_sched_complete(wubu_sched_t *s, int id);

/* Number of active (non-DONE) requests. */
int wubu_sched_active(const wubu_sched_t *s);

/* Allocate a request object. Caller fills tokens[] after. */
wubu_req_t *wubu_req_create(int id, const int *tokens, int n_prompt,
                            int prefix_len);
void wubu_req_free(wubu_req_t *r);

/* Emit a token for a request (append to its sequence). */
void wubu_req_emit(wubu_req_t *r, int token_id);

/* Compute the 64-bit hash of a token sequence (for prefix cache). */
uint64_t wubu_prefix_hash(const int *tokens, int n);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_SCHEDULER_H */
