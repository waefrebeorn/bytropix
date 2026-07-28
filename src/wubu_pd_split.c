/*
 * wubu_pd_split.c — Prefill/Decode disaggregation planner (Round-2 #131/#132).
 * C11, self-contained. Splits inference into a compute-bound prefill pool and a
 * bandwidth-bound decode pool (DistServe/Mooncake), and models the KV-cache
 * handoff over an RDMA-like transport. The actual RDMA verbs are environment-
 * specific; this module owns the *policy + transfer accounting* so the engine
 * can schedule phases independently and compute KV-transfer cost.
 */
#include "wubu_pd_split.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

struct wubu_pd_split {
    int n_prefill_gpus;
    int n_decode_gpus;
    double rdma_tb_s;     /* KV transfer bandwidth (TB/s): InfiniBand ~0.05, NVLink ~0.9 */
};

wubu_pd_split_t *wubu_pd_split_create(int np, int nd, double rdma_tb_s) {
    wubu_pd_split_t *s = (wubu_pd_split_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->n_prefill_gpus = np; s->n_decode_gpus = nd; s->rdma_tb_s = rdma_tb_s;
    return s;
}
void wubu_pd_split_free(wubu_pd_split_t *s) { free(s); }

/* KV transfer time (ms) for a prefix of `s_tokens` tokens at `layers` layers,
 * `kv_heads` GQA heads, `head_dim`, `kv_bits`. */
double wubu_pd_kv_transfer_ms(const wubu_pd_split_t *s, int s_tokens,
                              int layers, int kv_heads, int head_dim, int kv_bits) {
    if (s->rdma_tb_s <= 0) return 0;   /* DA: div-by-zero guard */
    double bytes = (double)s_tokens * layers * 2.0 * kv_heads * head_dim * (kv_bits/8.0);
    double sec = bytes / (s->rdma_tb_s * 1e12);
    return sec * 1000.0;
}

/* Route a request: pick the decode GPU with the smallest queue (load balance).
 * queues[] has n_decode_gpus entries (current queued token counts). */
int wubu_pd_route_decode(const wubu_pd_split_t *s, const int *queues) {
    int best = 0;
    int best_q = queues[0];
    for (int i = 1; i < s->n_decode_gpus; i++)
        if (queues[i] < best_q) { best_q = queues[i]; best = i; }
    return best;
}

/* Choose transfer mode: read-mode (decode pulls) when decode is the bottleneck
 * consumer; write-mode (prefill pushes) when prefill is compute-idle. Returns 1
 * for read-mode, 0 for write-mode. */
int wubu_pd_transfer_mode(const wubu_pd_split_t *s, double prefill_load,
                          double decode_load) {
    return decode_load >= prefill_load;
}
