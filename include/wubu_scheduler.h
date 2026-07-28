#ifndef WUBU_SCHEDULER_H
#define WUBU_SCHEDULER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { WUBU_REQ_PREFILL, WUBU_REQ_DECODE, WUBU_REQ_DONE } wubu_req_state;

typedef struct {
    int id;
    int *prompt;
    int prompt_len;
    int max_tokens;
    int *tokens;        /* emitted tokens */
    int n_kv;           /* total tokens processed (prompt + generated) */
    int n_gen;          /* generated count */
    uint64_t prefix_hash;
    int prefix_len_cache; /* how many leading prompt tokens are cacheable */
    int kv_reused;      /* 1 if prefix cache hit */
    wubu_req_state state;
} wubu_req_t;

wubu_req_t *wubu_req_create(int id, const int *prompt, int prompt_len, int max_tokens);
void wubu_req_free(wubu_req_t *r);
void wubu_req_emit(wubu_req_t *r, int tok);

typedef struct {
    uint64_t hash;
    int owner_id;
} wubu_pcache_t;

typedef struct {
    wubu_req_t **reqs;
    int n, cap;
    int max_batch;
    wubu_pcache_t *pcache;
    int pcache_n, pcache_cap;
} wubu_sched_t;

wubu_sched_t *wubu_sched_create(int max_batch);
void wubu_sched_free(wubu_sched_t *s);
int wubu_sched_submit(wubu_sched_t *s, wubu_req_t *r);
int wubu_sched_step(wubu_sched_t *s);   /* one decode iteration; returns active count */

uint64_t wubu_prefix_hash(const int *toks, int n);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SCHEDULER_H */
