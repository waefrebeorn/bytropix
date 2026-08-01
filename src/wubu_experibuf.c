/*
 * wubu_replay.c -- Experience replay buffer (BB01). C11.
 *
 * Convergence (experience replay + reservoir sampling 7-hop):
 *   - BB01: reservoir-sampled ring buffer of past sweep configurations.
 *     Each transition: params[15] + KV tag + tok_s + loss + oom_safe.
 *     Reservoir sampling gives unbiased representation of the infinite
 *     stream in O(1) per-item — perfect for memory-bounded at-home use.
 */
#include "wubu_experibuf.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

int wubu_experibuf_init(wubu_experibuf_t *r) {
    if (!r) return -1;
    r->count = 0;
    r->n = 0;
    return 0;
}

int wubu_experibuf_add(wubu_experibuf_t *r, const double *params, int ndim,
                    const char *kv_tag, double tok_s, double loss, int oom_safe) {
    if (!r || !params || ndim <= 0 || ndim > WUBU_REPLAY_DIMS) return -1;
    r->count++;
    unsigned idx;
    if (r->n < WUBU_REPLAY_CAPACITY) {
        idx = r->n;
        r->n++;
    } else {
        /* Reservoir sampling: replace element idx with probability capacity/count */
        double p = (double)WUBU_REPLAY_CAPACITY / (double)r->count;
        if ((double)rand() / (double)RAND_MAX > p) {
            return 0;  /* randomly drop */
        }
        idx = (unsigned)(rand() % WUBU_REPLAY_CAPACITY);
    }
    wubu_transition_t *t = &r->buf[idx];
    for (int i = 0; i < ndim; i++) t->params[i] = params[i];
    for (int i = ndim; i < WUBU_REPLAY_DIMS; i++) t->params[i] = 0.0;
    if (kv_tag) {
        snprintf(t->kv_tags, sizeof(t->kv_tags), "%s", kv_tag);
    } else {
        t->kv_tags[0] = '\0';
    }
    t->tok_s = tok_s;
    t->loss = loss;
    t->oom_safe = oom_safe ? 1 : 0;
    return 0;
}

int wubu_experibuf_sample(const wubu_experibuf_t *r, unsigned seed,
                       wubu_transition_t *out) {
    if (!r || !out || r->n == 0) return -1;
    /* Deterministic LCG for reproducible sampling */
    unsigned state = seed ? seed : 1;
    state = state * 1103515245U + 12345U;
    unsigned idx = state % r->n;
    *out = r->buf[idx];
    return 0;
}

unsigned wubu_experibuf_size(const wubu_experibuf_t *r) {
    return r ? r->n : 0;
}