/*
 * wubu_replay.h -- Experience replay buffer (BB01). C11.
 */
#ifndef WUBU_REPLAY_H
#define WUBU_REPLAY_H

#define WUBU_REPLAY_CAPACITY 128U
#define WUBU_REPLAY_DIMS 15
#define WUBU_REPLAY_KV_LEN 32

typedef struct {
    double params[WUBU_REPLAY_DIMS];   /* sweep dimensions */
    char kv_tags[WUBU_REPLAY_KV_LEN];  /* KV config tag (e.g. "512K-SRT") */
    double tok_s;                       /* observed throughput */
    double loss;                        /* observed loss (0 if unmeasured) */
    int    oom_safe;                    /* 1 if ran OOM-safe */
} wubu_transition_t;

typedef struct {
    wubu_transition_t buf[WUBU_REPLAY_CAPACITY];
    unsigned count;     /* total transitions seen (for reservoir) */
    unsigned n;         /* current fill level (≤ capacity) */
} wubu_experibuf_t;

int  wubu_experibuf_init(wubu_experibuf_t *r);
int  wubu_experibuf_add(wubu_experibuf_t *r, const double *params, int ndim,
                     const char *kv_tag, double tok_s, double loss, int oom_safe);
int  wubu_experibuf_sample(const wubu_experibuf_t *r, unsigned seed,
                        wubu_transition_t *out);
unsigned wubu_experibuf_size(const wubu_experibuf_t *r);

#endif