/*
 * wubu_moe_grouped.c — MoE expert grouping / grouped-GEMM dispatch
 * (Area D, items D.31/D.37/D.38). C11, self-contained.
 *
 * Given routing: token->expert assignments, group tokens per expert so a single
 * grouped GEMM (or one GEMM per expert over its token subset) runs efficiently.
 * Also exposes LRU expert hotness stats for redundant-expert placement (D.33).
 */
#include "wubu_moe_grouped.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

struct wubu_moe_router {
    int n_experts;
    int *count;          /* tokens routed to each expert */
    int **idx;           /* per-expert token index lists */
    int *hotness;        /* access count per expert (for D.33) */
};

wubu_moe_router_t *wubu_moe_router_create(int n_experts) {
    wubu_moe_router_t *r = (wubu_moe_router_t *)calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->n_experts = n_experts;
    r->count = (int *)calloc(n_experts, sizeof(int));
    r->idx = (int **)calloc(n_experts, sizeof(int *));
    r->hotness = (int *)calloc(n_experts, sizeof(int));
    for (int e = 0; e < n_experts; e++)
        r->idx[e] = (int *)malloc(sizeof(int) * 4096); /* generous per-expert cap */
    return r;
}
void wubu_moe_router_free(wubu_moe_router_t *r) {
    if (!r) return;
    for (int e = 0; e < r->n_experts; e++) free(r->idx[e]);
    free(r->idx); free(r->count); free(r->hotness);
    free(r);
}

/* Assign tokens to experts. routes[t] = expert id for token t (0..n_experts-1). */
void wubu_moe_router_assign(wubu_moe_router_t *r, const int *routes, int n_tokens) {
    for (int e = 0; e < r->n_experts; e++) r->count[e] = 0;
    for (int t = 0; t < n_tokens; t++) {
        int e = routes[t];
        assert(e >= 0 && e < r->n_experts);
        r->idx[e][r->count[e]++] = t;
        r->hotness[e]++;
    }
}

/* Return the top-k hottest experts (for redundant replication, D.33). */
void wubu_moe_router_top_hot(wubu_moe_router_t *r, int k, int *out_experts) {
    for (int i = 0; i < k; i++) out_experts[i] = -1;
    for (int i = 0; i < k; i++) {
        int best = -1, best_h = -1;
        for (int e = 0; e < r->n_experts; e++) {
            int taken = 0;
            for (int j = 0; j < i; j++) if (out_experts[j] == e) taken = 1;
            if (!taken && r->hotness[e] > best_h) { best_h = r->hotness[e]; best = e; }
        }
        out_experts[i] = best;
    }
}
int wubu_moe_router_count(wubu_moe_router_t *r, int e) { return r->count[e]; }
int *wubu_moe_router_idx(wubu_moe_router_t *r, int e) { return r->idx[e]; }
